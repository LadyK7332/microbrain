from controller import Robot

def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

robot = Robot()
timestep = int(robot.getBasicTimeStep())

print("mb_controller online ✅")

# Collect motors + limits
motors = {}  # name -> (motor, min, max)
count = robot.getNumberOfDevices()
for i in range(count):
    m = robot.getDeviceByIndex(i)
    name = m.getName()
    mn = m.getMinPosition()
    mx = m.getMaxPosition()
    motors[name] = (m, mn, mx)

# A simple "neutral-ish" pose within your limits
# (Goal: stop instant ragdoll and avoid illegal angles.)
targets = {
    # Arms: keep elbows at min (0) so no warnings, slight shoulder symmetry
    "LArmEly": 0.0,
    "LArmElx": 0.0,
    "RArmEly": 0.0,
    "RArmElx": 0.0,

    "LArmUsy": 0.0,
    "LArmShx": 0.0,
    "LArmUwy": 0.0,
    "LArmMwx": 0.0,

    "RArmUsy": 0.0,
    "RArmShx": 0.0,
    "RArmUwy": 0.0,
    "RArmMwx": 0.0,

    # Back/neck: neutral
    "BackLbz": 0.0,
    "BackMby": 0.0,
    "BackUbx": 0.0,
    "NeckAy": 0.0,

    # Legs: slight knee bend (knees must be >= 0)
    "LLegKny": 0.3,
    "RLegKny": 0.3,

    # Keep other leg joints neutral to start
    "LLegUhz": 0.0,
    "LLegMhx": 0.0,
    "LLegLhy": -0.3,
    "LLegUay": 0.0,
    "LLegLax": 0.0,

    "RLegUhz": 0.0,
    "RLegMhx": 0.0,
    "RLegLhy": -0.3,
    "RLegUay": 0.0,
    "RLegLax": 0.0,
}

# Main loop: hold the pose
while robot.step(timestep) != -1:
    for name, (m, mn, mx) in motors.items():
        desired = targets.get(name, 0.0)          # default to 0.0 if not specified
        m.setPosition(clamp(desired, mn, mx))
