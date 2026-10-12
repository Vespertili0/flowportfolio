def validate_weight(weight: float) -> None:
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise TypeError("Weight must be numeric.")
    if not (0.0 <= weight <= 1.0):
        raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")


def validate_fee(fee: float) -> None:
    if not isinstance(fee, (int, float)) or isinstance(fee, bool):
        raise TypeError("Fee must be numeric.")
    if fee < 0.0 or fee > 1.0:
        raise ValueError(f"Fee must be between 0.0 and 1.0, got {fee}")
