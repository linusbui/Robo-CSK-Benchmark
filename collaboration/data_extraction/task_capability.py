from gripper_configs import GripperConfig

class TaskCapability:
    def __init__(self, task: str, mobile: bool, no_arms: int, arm_dof: int, gripper_conf: GripperConfig, rigid_gripper: bool):
        self._tsk = preprocess_string(task)
        self._mobile = mobile
        self._arms = no_arms
        self._arm_dof = arm_dof
        self._gripper_conf = gripper_conf
        self._rigid_gripper = rigid_gripper
        self._is_correct = True

    def __str__(self):
        if self._mobile:
            walk = 'does need to walk'
        else:
            walk = 'does NOT need to walk'

        if self._rigid_gripper:
            rigidity = 'rigid'
        else:
            rigidity = 'soft'

        return (f'To {self._tsk} the robot needs {self._arms} arm(s) with {self._arm_dof} DoFs and {rigidity} {self._gripper_conf.lower()}.'
                f' It {walk}.')

    def verify(self) -> bool:
        return self._is_correct

    def get_task(self) -> str:
        return self._tsk

    def is_mobile(self) -> bool:
        return self._mobile

    def get_arms(self) -> int:
        return self._arms

    def get_arm_dofs(self) -> int:
        return self._arm_dof

    def get_gripper_config(self) -> GripperConfig:
        return self._gripper_conf

    def is_rigid_gripper(self) -> bool:
        return self._rigid_gripper

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
            'Mobile?': self.is_mobile(),
            'Arms': self.get_arms(),
            'DoFs': self.get_arm_dofs(),
            'Gripper Config': self.get_gripper_config(),
            'Rigid Gripper?': self.is_rigid_gripper()
        }


def preprocess_string(word: str) -> str:
    return word.lower().replace('_', ' ').replace('\"', "").strip()


def combine_all_tasks(caps: [TaskCapability]) -> [TaskCapability]:
    combined = {}
    for tup in [t for t in caps if t.verify]:
        obj = tup.get_task()
        if obj in combined:
            combined[obj].combine_task(tup)
        else:
            combined[obj] = tup
    return list(combined.values())
