#-*- coding: utf-8 -*-


class DecoderError(RuntimeError):
    pass

class DecoderNotFoundError(DecoderError):
    pass


format2decoder_map = {}
name2decoder_map = {}
decoder_names = ['audioread']
for name in decoder_names:
    try:
        model = __import__('_dc_'+name)
        for fmt in model.support_formats:
            arr = format2decoder_map.setdefault(fmt,[])
            arr.append(model)
        name2decoder_map[name] = model
    except ImportError as ex:
        print ex


def get_decoder(format=None, name=None):
    assert format or name
    format = format.lower()
    if format:
        if not name:
            arr = format2decoder_map.get(format, [])
            if not arr:
                raise DecoderNotFoundError
            return arr[0]
        else:
            model = name2decoder_map.get(name, None)
            if model is None or format not in model.support_formats:
                raise DecoderNotFoundError
            return model
    elif name:
        model = name2decoder_map.get(name, None)
        if model is None:
            raise DecoderNotFoundError
        return model

