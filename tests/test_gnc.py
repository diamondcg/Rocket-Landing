"""Unit and integration tests for the Rocket Landing GNC modules."""

import sys
import os

import numpy as np
import pytest

# Ensure the project root is on the path so imports work when tests are run
# from any directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dynamics import RocketDynamics
from controller import PIDController
from estimator import KalmanFilter
from simulation import load_config, run_simulation
from monte_carlo import run_monte_carlo
from thermal import ThermalModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    return load_config(cfg_path)


# ===========================================================================
# dynamics.py
# ===========================================================================

class TestRocketDynamics:
    def test_free_fall_no_thrust(self):
        """With zero thrust the rocket should accelerate downward."""
        dyn = RocketDynamics(gravity=9.81, alpha=0.0005, dt=0.01)
        state = np.array([100.0, 0.0, 20.0])
        next_state = dyn.step(state, thrust=0.0)
        # altitude should decrease (velocity becomes negative)
        assert next_state[1] < state[1], "Velocity should decrease under gravity"
        assert next_state[0] < state[0] or np.isclose(next_state[0], state[0], atol=1e-3)

    def test_hover_thrust(self):
        """With thrust equal to m*g the rocket should not accelerate."""
        dyn = RocketDynamics(gravity=9.81, alpha=0.0, dt=0.01)  # alpha=0 for simplicity
        m = 20.0
        hover_thrust = m * 9.81
        state = np.array([100.0, 0.0, m])
        next_state = dyn.step(state, thrust=hover_thrust)
        assert abs(next_state[1]) < 1e-9, "Velocity should stay zero at hover thrust"

    def test_mass_decreases_with_thrust(self):
        """Mass should decrease when thrust is positive."""
        dyn = RocketDynamics(gravity=9.81, alpha=0.0005, dt=0.01)
        state = np.array([100.0, 0.0, 20.0])
        next_state = dyn.step(state, thrust=100.0)
        assert next_state[2] < state[2], "Mass should decrease with thrust"

    def test_mass_non_negative(self):
        """Mass should never go below 1e-6."""
        dyn = RocketDynamics(gravity=9.81, alpha=1.0, dt=1.0)  # extreme alpha
        state = np.array([100.0, 0.0, 0.0])  # zero mass
        next_state = dyn.step(state, thrust=1000.0)
        assert next_state[2] >= 1e-6

    def test_state_shape(self):
        dyn = RocketDynamics()
        state = np.array([50.0, -5.0, 15.0])
        next_state = dyn.step(state, thrust=200.0)
        assert next_state.shape == (3,)


# ===========================================================================
# controller.py
# ===========================================================================

class TestPIDController:
    def test_output_within_bounds(self):
        """Controller output must always stay within thrust bounds."""
        ctrl = PIDController(Kp=15.0, Ki=2.0, Kd=10.0,
                             thrust_min=0.0, thrust_max=500.0, dt=0.01)
        for z in [100.0, 50.0, 10.0, 1.0, 0.01, -1.0]:
            state = np.array([z, -5.0, 20.0])
            u = ctrl.compute(state)
            assert 0.0 <= u <= 500.0, f"Thrust {u} out of bounds for z={z}"

    def test_reset_clears_integral(self):
        ctrl = PIDController(Kp=5.0, Ki=1.0, Kd=0.0,
                             thrust_min=0.0, thrust_max=1000.0, dt=0.01)
        state = np.array([100.0, 0.0, 20.0])
        for _ in range(100):
            ctrl.compute(state)
        ctrl.reset()
        assert ctrl._integral == 0.0
        assert ctrl._first_step is True

    def test_positive_altitude_produces_thrust(self):
        """Rocket above ground (positive z) should receive upward thrust."""
        cfg = _default_cfg()
        pid = cfg["controller"]["pid"]
        ctrl = PIDController(
            Kp=pid["Kp"], Ki=pid["Ki"], Kd=pid["Kd"],
            thrust_min=cfg["rocket"]["thrust_min"],
            thrust_max=cfg["rocket"]["thrust_max"],
            dt=cfg["simulation"]["dt"],
        )
        state = np.array([50.0, 0.0, 20.0])
        u = ctrl.compute(state)
        assert u > 0.0, "Should produce positive thrust when above ground"


# ===========================================================================
# estimator.py
# ===========================================================================

class TestKalmanFilter:
    def _make_kf(self):
        dt = 0.01
        Q = np.diag([0.01, 0.01, 0.0])
        R = 1.0
        P_init = np.diag([10.0, 1.0, 0.1])
        kf = KalmanFilter(dt=dt, Q=Q, R=R, P_init=P_init)
        kf.initialise(np.array([100.0, -5.0, 20.0]))
        return kf

    def test_state_property_returns_copy(self):
        kf = self._make_kf()
        s1 = kf.state
        s2 = kf.state
        s1[0] = 999.0
        assert kf.state[0] != 999.0, "state property should return a copy"

    def test_predict_changes_state(self):
        kf = self._make_kf()
        before = kf.state.copy()
        kf.predict(thrust=200.0)
        after = kf.state
        assert not np.allclose(before, after), "Predict should update state estimate"

    def test_update_incorporates_measurement(self):
        kf = self._make_kf()
        kf.predict(thrust=200.0)
        state_after_predict = kf.state.copy()
        # measurement far from predicted position should pull estimate toward it
        kf.update(measurement=80.0)
        assert not np.allclose(kf.state, state_after_predict)

    def test_covariance_positive_definite(self):
        kf = self._make_kf()
        for _ in range(10):
            kf.predict(100.0)
            kf.update(kf.state[0] + 0.5)
        eigenvalues = np.linalg.eigvalsh(kf.P)
        assert np.all(eigenvalues >= 0), "Covariance must remain positive semi-definite"

    def test_converges_to_true_position(self):
        """KF estimate should converge toward the true position over many steps."""
        rng = np.random.default_rng(0)
        kf = self._make_kf()
        true_z = 100.0
        for _ in range(200):
            kf.predict(thrust=0.0)
            kf.update(true_z + rng.normal(0, 1.0))
        assert abs(kf.state[0] - true_z) < 2.0, "KF should converge near true position"


# ===========================================================================
# simulation.py
# ===========================================================================

class TestSimulation:
    def test_load_config(self):
        cfg = _default_cfg()
        assert "simulation" in cfg
        assert "rocket" in cfg
        assert "controller" in cfg
        assert "estimator" in cfg
        assert "noise" in cfg
        assert "monte_carlo" in cfg

    def test_single_run_returns_expected_keys(self):
        cfg = _default_cfg()
        result = run_simulation(cfg, seed=0)
        for key in ("time", "z", "v", "mass", "z_hat", "v_hat", "thrust",
                    "landed", "final_z"):
            assert key in result, f"Missing key: {key}"

    def test_simulation_lands(self):
        """Default simulation should land the rocket."""
        cfg = _default_cfg()
        result = run_simulation(cfg, seed=42)
        assert result["landed"], "Rocket should land within the simulation time"

    def test_landing_accuracy(self):
        """Final position should be within 0.1 m of ground."""
        cfg = _default_cfg()
        result = run_simulation(cfg, seed=42)
        assert abs(result["final_z"]) <= 0.1, (
            f"Landing error {abs(result['final_z']):.4f} m exceeds 0.1 m threshold"
        )

    def test_log_arrays_same_length(self):
        cfg = _default_cfg()
        result = run_simulation(cfg, seed=1)
        n = len(result["time"])
        for key in ("z", "v", "mass", "z_hat", "v_hat", "thrust"):
            assert len(result[key]) == n, f"Array '{key}' length mismatch"

    def test_thrust_within_bounds(self):
        cfg = _default_cfg()
        result = run_simulation(cfg, seed=2)
        thrust_min = cfg["rocket"]["thrust_min"]
        thrust_max = cfg["rocket"]["thrust_max"]
        assert np.all(result["thrust"] >= thrust_min)
        assert np.all(result["thrust"] <= thrust_max)


# ===========================================================================
# monte_carlo.py
# ===========================================================================

class TestMonteCarlo:
    def test_mc_returns_expected_keys(self):
        cfg = _default_cfg()
        results = run_monte_carlo(cfg, n_runs=10, seed=0)
        for key in ("n_runs", "success_count", "failed_count",
                    "success_rate", "final_z_errors", "mean_error",
                    "std_error", "max_error"):
            assert key in results

    def test_mc_run_count(self):
        cfg = _default_cfg()
        results = run_monte_carlo(cfg, n_runs=20, seed=1)
        assert results["n_runs"] == 20
        assert len(results["final_z_errors"]) == 20

    def test_mc_success_rate_reasonable(self):
        """Success rate should be > 0 for reasonable initial conditions."""
        cfg = _default_cfg()
        results = run_monte_carlo(cfg, n_runs=50, seed=7)
        assert results["success_rate"] >= 0.0
        assert results["success_rate"] <= 1.0

    def test_mc_1000_runs_success_rate(self):
        """With 1000 runs the success rate should be high (> 80%)."""
        cfg = _default_cfg()
        results = run_monte_carlo(cfg, n_runs=1000, seed=99)
        assert results["success_rate"] >= 0.8, (
            f"Monte Carlo success rate {results['success_rate']*100:.1f}% < 80%"
        )


# ===========================================================================
# thermal.py
# ===========================================================================

class TestThermalModel:
    def _model(self, cfg=None):
        cfg = cfg or _default_cfg()
        thermal_cfg = cfg["thermal"]
        return ThermalModel(thermal_cfg["segment_params"],
                             thermal_cfg["initial_temps"])

    def test_initial_temps_match_config(self):
        cfg = _default_cfg()
        model = self._model(cfg)
        assert model.temps == cfg["thermal"]["initial_temps"]

    def test_reset_restores_initial_temps(self):
        cfg = _default_cfg()
        model = self._model(cfg)
        for _ in range(10):
            model.step(v=-50.0, thrust=500.0, thrust_max=500.0,
                       ambient_temp=288.0, humidity=50.0, dt=0.1)
        model.reset()
        assert model.temps == cfg["thermal"]["initial_temps"]

    def test_step_returns_copy_not_reference(self):
        model = self._model()
        result = model.step(v=-10.0, thrust=0.0, thrust_max=500.0,
                             ambient_temp=288.0, humidity=50.0, dt=0.1)
        result["forward"] = -1.0
        assert model.temps["forward"] != -1.0

    def test_aero_heating_increases_with_velocity(self):
        cfg = _default_cfg()
        model_slow = self._model(cfg)
        model_fast = self._model(cfg)
        ambient = cfg["thermal"]["initial_temps"]["forward"]

        slow = model_slow.step(v=-5.0, thrust=0.0, thrust_max=500.0,
                                ambient_temp=ambient, humidity=50.0, dt=0.1)
        fast = model_fast.step(v=-50.0, thrust=0.0, thrust_max=500.0,
                                ambient_temp=ambient, humidity=50.0, dt=0.1)
        assert fast["forward"] > slow["forward"]

    def test_forward_has_highest_aero_coefficient(self):
        cfg = _default_cfg()
        params = cfg["thermal"]["segment_params"]
        assert params["forward"]["aero_coeff"] > params["mid"]["aero_coeff"]
        assert params["forward"]["aero_coeff"] > params["aft"]["aero_coeff"]

    def test_aft_heats_with_thrust(self):
        cfg = _default_cfg()
        model = self._model(cfg)
        ambient = cfg["thermal"]["initial_temps"]["aft"]

        result = model.step(v=0.0, thrust=500.0, thrust_max=500.0,
                             ambient_temp=ambient, humidity=50.0, dt=0.1)
        assert result["aft"] > ambient
        # forward/mid have engine_coeff == 0 and v == 0 -> no heating
        assert result["forward"] <= ambient
        assert result["mid"] <= ambient

    def test_cooling_toward_ambient(self):
        cfg = _default_cfg()
        params = cfg["thermal"]["segment_params"]
        initial_temps = {seg: 500.0 for seg in cfg["thermal"]["initial_temps"]}
        model = ThermalModel(params, initial_temps)
        ambient = 288.0

        prev = dict(model.temps)
        for _ in range(50):
            result = model.step(v=0.0, thrust=0.0, thrust_max=500.0,
                                 ambient_temp=ambient, humidity=50.0, dt=0.05)
            for seg in result:
                assert result[seg] < prev[seg]
                assert result[seg] > ambient
            prev = result

    def test_humidity_increases_cooling_rate(self):
        cfg = _default_cfg()
        params = cfg["thermal"]["segment_params"]
        initial_temps = {seg: 500.0 for seg in cfg["thermal"]["initial_temps"]}
        ambient = 288.0

        model_dry = ThermalModel(params, dict(initial_temps))
        model_humid = ThermalModel(params, dict(initial_temps))
        for _ in range(20):
            dry = model_dry.step(v=0.0, thrust=0.0, thrust_max=500.0,
                                  ambient_temp=ambient, humidity=0.0, dt=0.05)
            humid = model_humid.step(v=0.0, thrust=0.0, thrust_max=500.0,
                                      ambient_temp=ambient, humidity=100.0, dt=0.05)
        for seg in dry:
            assert humid[seg] < dry[seg]

    def test_step_with_zero_dt_no_change(self):
        cfg = _default_cfg()
        model = self._model(cfg)
        before = dict(model.temps)
        after = model.step(v=-50.0, thrust=500.0, thrust_max=500.0,
                            ambient_temp=200.0, humidity=50.0, dt=0.0)
        assert after == before
