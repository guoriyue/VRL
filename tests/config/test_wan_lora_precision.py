from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from vrl.config.rules import check_cross_section_rules
from vrl.models.families.wan_2_1.config import WanModelSection


def test_fp32_adapter_schema_requires_lora() -> None:
    with pytest.raises(ValidationError, match=r"requires model\.use_lora"):
        WanModelSection(family="wan_2_1", use_lora=False, lora_parameter_dtype="float32")
    model = WanModelSection(family="wan_2_1", use_lora=True, lora_parameter_dtype="float32")
    assert model.lora_parameter_dtype == "float32"
    assert WanModelSection(family="wan_2_1").lora_parameter_dtype is None


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "auto"])
def test_adapter_schema_rejects_unsupported_dtype(dtype: str) -> None:
    with pytest.raises(ValidationError, match="lora_parameter_dtype"):
        WanModelSection(family="wan_2_1", use_lora=True, lora_parameter_dtype=dtype)


@pytest.mark.parametrize("policy", [None, "actor", "none"])
@pytest.mark.parametrize("dtype", [None, "float32"])
def test_fp32_adapter_requires_preserving_fsdp_policy(
    policy: str | None,
    dtype: str | None,
) -> None:
    root = SimpleNamespace(
        model=SimpleNamespace(lora_parameter_dtype=dtype),
        algorithm=None,
        distributed=SimpleNamespace(
            training=SimpleNamespace(
                strategy="fsdp",
                fsdp=None if policy is None else SimpleNamespace(precision_policy=policy),
            )
        ),
    )
    if dtype == "float32" and policy != "none":
        with pytest.raises(ValueError, match="precision_policy=none"):
            check_cross_section_rules(root)
    else:
        check_cross_section_rules(root)
