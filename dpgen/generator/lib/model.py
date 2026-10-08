"""Model-family detection shared by training and exploration."""


def is_dpa4c(jdata):
    """Recognize both foundation fine-tuning and explicit DPA4C descriptors."""
    if not jdata:
        return False
    descriptor = (
        jdata.get("default_training_param", {}).get("model", {}).get("descriptor", {})
    )
    return (
        jdata.get("finetune_model_type") == "dpa4c" or descriptor.get("type") == "dpa4c"
    )


def is_dpa4(jdata):
    """Recognize DPA4 fine-tuning and explicit DPA4/SeZM descriptors."""
    if not jdata:
        return False
    model = jdata.get("default_training_param", {}).get("model", {})
    return (
        jdata.get("finetune_model_type") == "dpa4"
        or model.get("type") in {"dpa4", "DPA4", "sezm", "SeZM"}
        or model.get("descriptor", {}).get("type") in {"dpa4", "sezm"}
    )


def uses_pt2(jdata):
    """Identify model families deployed as compiled PyTorch archives."""
    return is_dpa4(jdata) or is_dpa4c(jdata)


def prepare_training_backend(jdata):
    """Select the PyTorch backend for DPA4C without changing its model definition."""
    if is_dpa4c(jdata):
        if jdata.get("train_backend", "pytorch") != "pytorch":
            raise ValueError("DPA4C requires train_backend='pytorch'")
        jdata["train_backend"] = "pytorch"
        jdata["dp_train_skip_neighbor_stat"] = True
    return jdata
