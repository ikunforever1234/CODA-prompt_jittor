"""Learner implementations using Jittor."""

from . import base, prompt

LEARNER_MODULES = {
	'base': base,
	'default': base,
	'prompt': prompt,
}


def build_learner(learner_type: str, learner_name: str, config):
	module = LEARNER_MODULES.get(learner_type)
	if module is None or not hasattr(module, learner_name):
		raise ValueError(f'Unknown learner {learner_type}.{learner_name}')
	learner_cls = getattr(module, learner_name)
	return learner_cls(config)


__all__ = ['build_learner']
