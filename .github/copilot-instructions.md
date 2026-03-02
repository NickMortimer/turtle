---
description: 'Albatri UAV mission planning library guidelines'
---

# Albatri Project Instructions

Albatri is a **safety-critical** multi-gimbal UAV mission planning library for ArduCopter. It validates missions, computes flight paths and gimbal orientations, simulates coverage, and exports MAVLink waypoint files.

## Safety-First Mindset

This codebase plans real drone flights. Silent failures or permissive defaults can lead to crashes, lost equipment, or worse.

**Mandatory rules:**
- Never infer defaults for safety parameters (endurance, wind limits, speeds, slew rates). Raise `ConfigurationError` immediately.
- Use explicit comparisons (`if value is None`) not truthiness (`if not value`).
- Tests must assert that missing required fields raise errors.

**Triple-gate validation:**
1. **Load time** (`config.py`): Validate YAML structure, required fields, type constraints
2. **Pre-check** (`SafetyEngine.validate()`): Validate Mission object structure before rule execution
3. **Rule validation** (individual `SafetyRule.check()`): Domain-specific safety constraints

**Reference:** [docs/SAFETY_INVARIANTS.md](docs/SAFETY_INVARIANTS.md)

## Architecture Decisions (ADRs)

Before adding dependencies or changing patterns, consult [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md).

Key constraints:
- **ADR-001 DroneKit scope:** Use `dronekit.Command` only for MAVLink item parameters in Phase 5. No `Vehicle`, `connect()`, or runtime control.
- **ADR-002 Dataclasses:** Use stdlib `dataclasses` for models; no Pydantic. Validate at YAML boundary.
- **ADR-003 Typer CLI:** Multi-command CLI via Typer.
- **ADR-005 Geodetics:** Use `pyproj.Geod(ellps='WGS84')` for all distance/bearing calculations. No Haversine.
- **ADR-006 Camera triggers:** Autopilot-delegated triggering via `MAV_CMD_DO_SET_CAM_TRIGG_DIST`. Do not generate dense waypoint lists.
- **ADR-007 Gimbal frame:** ArduPilot body frame (X-fwd, Y-right, Z-down). Pitch negative = down.

## Module Responsibilities

| Module | Purpose | Key classes/functions |
|--------|---------|----------------------|
| `core/` | Models, config loading, constants | `Mission`, `load_mission()`, `ConfigurationError` |
| `safety/` | Rule engine, validation | `SafetyEngine`, `SafetyRule`, `ValidationReport` |
| `planning/` | Path generation, sampling | `plan_corridor()`, `plan_area_sweep()`, `Geodesic` |
| `gimbal/` | Sun position, orientation, glint | `compute_sun_position()`, `compute_gimbal_angles()` |
| `sim/` | Mission simulation, coverage | `SimulationEngine`, `SimulationEvent` |
| `mavlink/` | MAVLink export (Phase 5) | `build_mission_items()`, `export_waypoint_file()` |
| `cli/` | Typer CLI commands | `validate`, `simulate`, `export` |

## Coding Standards

### Style
- PEP 8, 79-char lines, 4-space indent.
- Type hints on all public functions.
- PEP 257 docstrings with Parameters/Returns sections.

### Patterns
- Dataclasses for data; explicit validation at YAML boundary.
- Raise `ConfigurationError` for invalid configs, `ValueError` for bad runtime arguments.
- Use `logging.getLogger(__name__)` per module.

### Key Dependencies
- **pyproj** (≥3.6.0): Geodetic calculations (WGS84 ellipsoid)
- **astral** (≥3.2): Sun position calculations
- **typer** (≥0.9.0): CLI framework
- **pymavlink** (≥2.4.39): MAVLink message generation

### Logging

All implementation must use the centralized logging infrastructure defined in `core/logging_config.py`. This ensures consistent message formatting and integrates with CLI verbosity flags.

**Logging usage pattern:**
```python
import logging
logger = logging.getLogger(__name__)

# INFO level: Sparse, high-level messages for CLI user experience
logger.info(f"Starting mission validation: {mission_name}")
logger.info("Flight path generated successfully")

# DEBUG level: Detailed output for complex/CPU-intensive operations
logger.debug(f"Validating wind constraint: current={wind_speed}, max={max_wind}")
logger.debug(f"Geodetic calculation: distance={dist:.2f}m, bearing={az:.1f}°")
```

**Guidelines:**
- **INFO**: User-facing progress updates, completion milestones, high-level results. Use sparingly for clean CLI output.
- **DEBUG**: Intermediate calculations, loop iterations, rule evaluations, complex algorithm steps, CPU-intensive operations.
- Never use `print()` in library code; always use logging.
- CLI commands initialize logging via `setup_logging()` from `core.logging_config`.
- Use hierarchical logger names (e.g., `logging.getLogger(__name__)`) for automatic `albatri.module.submodule` naming.

### Testing
- pytest with markers: `@pytest.mark.unit`, `@pytest.mark.integration`.
- Apply ISTQB techniques: equivalence partitioning, boundary value analysis, decision tables.
- Parameterize boundary/edge-case tests.
- Every safety rule needs at least one positive and one negative test.
- All tests must have docstrings explaining purpose.
- Test coverage ≥ 90%.

**Safety rule test pattern:**
```python
@pytest.mark.unit
def test_wind_speed_rule_pass():
    """Test wind speed rule passes when within limits."""
    # Arrange: mission with wind < max
    # Act: validate
    # Assert: passes = True

@pytest.mark.unit
def test_wind_speed_rule_fail():
    """Test wind speed rule fails when exceeding limits."""
    # Arrange: mission with wind > max
    # Act: validate
    # Assert: passes = False, error message clear
```

## Common Pitfalls (Do NOT Do)

❌ **Never use Haversine formula** - use `pyproj.Geod()` for geodetic calculations  
❌ **Never generate dense waypoint lists for camera triggers** - use `MAV_CMD_DO_SET_CAM_TRIGG_DIST`  
❌ **Never add Pydantic or validation frameworks** - use explicit validation in `config.py`  
❌ **Never assume antipodal points won't occur** - test geodetic edge cases  
❌ **Never use truthiness for safety checks** - explicit `is None` comparisons only

## Geodetic Edge Cases

Always handle:
- **Antipodal points** (opposite sides of Earth): `pyproj.Geod.inv()` may fail or return undefined azimuth
- **Dateline crossing** (longitude wraps at ±180°): ensure path logic handles wrap-around
- **Polar regions** (near ±90° latitude): convergence of meridians affects bearings
- **Zero-distance segments**: validate waypoint separation > epsilon (e.g., 0.1m)

Test framework validates these in `tests/planning/test_edge_cases.py`.

## Quick Reference

```python
# Loading a mission (safety-validated)
from albatri.core.config import load_mission
mission = load_mission("examples/missions/dawn_coastal_survey.yaml")

# Validating safety rules
from albatri.safety.rules import SafetyEngine
engine = SafetyEngine()
report = engine.validate(mission)

# Generating flight paths
from albatri.planning.paths import generate_flight_path
paths = generate_flight_path(mission)

# Geodesic calculations (always use pyproj)
from albatri.planning.geometry import Geodesic
geo = Geodesic()
distance = geo.distance(lat1, lon1, lat2, lon2)
```

## Development Workflow

### Building and Testing
- Run tests with: `pytest tests/` (use markers: `-m unit` or `-m integration`)
- Check coverage with: `pytest --cov=albatri --cov-report=term-missing`
- Lint code with: `ruff check src/` and `ruff format src/`
- Type check with: `mypy src/albatri/`

### Pre-commit Hooks
- Pre-commit hooks are configured in `.pre-commit-config.yaml`
- Run manually with: `pre-commit run --all-files`
- Install hooks with: `pre-commit install`

### Mission Files
- Example missions are in `examples/missions/`
- Mission schema follows YAML structure defined in `core/config.py`
- Always validate new mission files with the CLI: `python -m albatri.cli validate <mission.yaml>`

## Boundaries and Exclusions

**Never modify or delete:**
- Example mission files in `examples/missions/` (these are reference implementations)
- Safety validation rules without team review (safety-critical)
- MAVLink command definitions (must match ArduPilot specification)
- Existing ADR files in `docs/` (architectural decisions are immutable)

**Always validate before committing:**
- Mission YAML files must validate successfully
- All safety tests must pass
- Type hints must be complete and mypy must pass
- Code coverage must remain ≥ 90%

## File Organization

```
albatri/
├── src/albatri/          # Main package
│   ├── core/            # Mission models, config loading
│   ├── safety/          # Validation engine and rules
│   ├── planning/        # Path generation, geodetics
│   ├── gimbal/          # Sun position, gimbal control
│   ├── sim/             # Mission simulation
│   ├── mavlink/         # MAVLink export (Phase 5)
│   └── cli/             # Typer CLI commands
├── tests/               # Test suite (mirrors src structure)
├── examples/            # Example missions and usage
├── docs/                # Architecture decisions and documentation
└── plans/               # Implementation planning documents
```

## Specialized Agents

For specific tasks, use the appropriate agent mode:
- **Thinking-Beast-Mode**: Complex problems requiring deep research and multi-step reasoning
- **blueprint-mode**: Structured workflows with strict validation (Debug, Express, Main, Loop)

See [AGENTS.md](../AGENTS.md) for full agent descriptions.

## Additional Resources

- [Safety Invariants](docs/SAFETY_INVARIANTS.md): Required mission parameters and validation gates
- [Architecture Decisions](docs/ARCHITECTURE_DECISIONS.md): Design rationale and technical constraints
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md): Development phases and milestones
