def profit_percentage(buy_price: float, sell_price: float) -> float:
    """Calculate profit percentage from buy and sell prices."""
    return (sell_price - buy_price) / buy_price

def expected_return_curve(current_price: float, target_return: float) -> float:
    """Solve for the price required to exit with a target return."""
    return current_price * (1 + target_return) 