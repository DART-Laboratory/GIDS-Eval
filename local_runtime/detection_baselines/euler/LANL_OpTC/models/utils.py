import os

import torch.distributed.rpc as rpc

def _local_mode_enabled() -> bool:
    raw = os.environ.get("EG1_EULER_LOCAL_MODE", "")
    return raw.strip().lower() not in {"", "0", "false", "no"}

LOCAL_MODE = _local_mode_enabled()

class _ImmediateFuture:
    def __init__(self, value=None, exc: Exception | None = None):
        self._value = value
        self._exc = exc

    def wait(self):
        if self._exc is not None:
            raise self._exc
        return self._value

class LocalRRef:
    def __init__(self, value):
        self._value = value

    def local_value(self):
        return self._value

    def owner(self):
        return None

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    if LOCAL_MODE:
        return _call_method(method, rref, *args, **kwargs)
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs, timeout=0.0)

def _remote_method_async(method, rref, *args, **kwargs):
    if LOCAL_MODE:
        try:
            return _ImmediateFuture(_call_method(method, rref, *args, **kwargs))
        except Exception as exc:
            return _ImmediateFuture(exc=exc)
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs, timeout=0.0)

def _param_rrefs(module):
    if LOCAL_MODE:
        return list(module.parameters())
    rrefs = []
    for param in module.parameters():
        rrefs.append(
            rpc.RRef(param)
        )

    return rrefs

def _decoder_rrefs(module):
    if LOCAL_MODE:
        return list(module.decoder_parameters())
    rrefs = []
    for param in module.decoder_parameters():
        rrefs.append(
            rpc.RRef(param)
        )

    return rrefs
