"""Time step management (if discrete stepping is needed)."""

class TimeStepManager:
    def __init__(self, env, step_minutes=1):
        """
        Manage simulation time in fixed increments (optional).
        """
        self.env = env
        self.step = step_minutes
    
    def step_forward(self):
        """
        Advance the simulation by one time step.
        """
        current_time = self.env.now
        target_time = current_time + self.step
        self.env.run(until=target_time)