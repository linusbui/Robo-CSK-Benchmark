class TaskCapability:
    def __init__(self, task: str, no_arms: int, arm_dof: int, mobile: bool):
        self._tsk = preprocess_string(task)
        self._arms = no_arms
        self._arm_dof = arm_dof
        self._mobile = mobile
        self._is_correct = True

    def __str__(self):
        if self._mobile:
            walk = 'does need to walk'
        else:
            walk = 'does NOT need to walk'
        return f'To {self._tsk} the robot needs {self._arms} arm(s) with {self._arm_dof} DoFs and {walk}'

    def verify(self) -> bool:
        return self._is_correct

    def get_task(self) -> str:
        return self._tsk

    def get_arms(self) -> int:
        return self._arms

    def get_arm_dofs(self) -> int:
        return self._arm_dof

    def is_mobile(self) -> bool:
        return self._mobile

    def combine_task(self, cap: 'TaskCapability'):
        if not cap.verify():
            return

        self._arms = min(cap.get_arms(), self._arms)
        self._arm_dof = min(cap.get_arm_dofs(), self._arm_dof)
        self._mobile = cap.is_mobile() and self._mobile
        cap._is_correct = False

    def to_dict(self):
        return {
            'Task': self.get_task(),
            'Arms': self.get_arms(),
            'DoFs': self.get_arm_dofs(),
            'Mobile?': self.is_mobile()
        }


def preprocess_string(word: str) -> str:
    return word.lower().replace('_', ' ').strip()


def combine_all_tasks(caps: [TaskCapability]) -> [TaskCapability]:
    combined = {}
    for tup in [t for t in caps if t.verify]:
        obj = tup.get_task()
        if obj in combined:
            combined[obj].combine_task(tup)
        else:
            combined[obj] = tup
    return list(combined.values())
