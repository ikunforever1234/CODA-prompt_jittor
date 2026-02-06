"""Model definitions backed by Jittor."""

from . import vit, zoo

MODEL_TYPES = {
	'vit': vit,
	'zoo': zoo,
}


def get_model(model_type: str, model_name: str):
	module = MODEL_TYPES.get(model_type)
	if module is None or not hasattr(module, model_name):
		raise ValueError(f'Unknown model {model_type}.{model_name}')
	return getattr(module, model_name)


__all__ = ['get_model']
