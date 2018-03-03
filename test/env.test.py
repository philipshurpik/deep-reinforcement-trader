import unittest
import sys
sys.path.append('..')
from market_env import MarketEnv

DECIMAL_SIGNS = 6
rnd = lambda x: round(x, DECIMAL_SIGNS)


class TestMarketEnv(unittest.TestCase):
    def setUp(self):
        self.env = MarketEnv(csv_name="../data/BTCETH60.csv", window_size=10, seed=42)
        self.calc = self.env._calculate_step

    """
    Hold action
    """
    def test_calculate_step_hold(self):
        amount = -0.5
        do_action = -0.1
        action = [amount, do_action]
        result = self.calc(action=action, close_price=0.1, position_value=0, position_size=0, remaining_cash=1)
        self.assertEqual(result, {"value": 0, "value_diff": 0, "size": 0, "size_diff": 0, "remaining_cash": 1, "reward": 0, "total_cash": 1})

    def test_calculate_step_hold_exist_position_no_price_change(self):
        amount = -0.5
        do_action = -0.1
        action = [amount, do_action]
        result = self.calc(action=action, close_price=0.1, position_value=0.5, position_size=5, remaining_cash=1)
        self.assertEqual(result, {"value": 0.5, "value_diff": 0, "size": 5, "size_diff": 0, "remaining_cash": 1, "reward": 0, "total_cash": 1.5})

    def test_calculate_step_hold_exist_long_position_with_positive_price_change(self):
        action = [1, -0.1]
        result = self.calc(action=action, close_price=0.25, position_value=0.5, position_size=5, remaining_cash=1)
        self.assertEqual(result, {"value": 1.25, "value_diff": 0.75, "size": 5, "size_diff": 0, "remaining_cash": 1, "reward": 0.75, "total_cash": 2.25})

    def test_calculate_step_hold_exist_long_position_with_negative_price_change(self):
        action = [1, -0.1]
        result = self.calc(action=action, close_price=0.075, position_value=0.5, position_size=5, remaining_cash=1)
        self.assertEqual(result, {"value": 0.375, "value_diff": -0.125, "size": 5, "size_diff": 0, "remaining_cash": 1, "reward": -0.125, "total_cash": 1.375})

    def test_calculate_step_hold_exist_short_position_with_positive_price_change(self):
        action = [1, -0.1]
        result = self.calc(action=action, close_price=3.5, position_value=-6, position_size=-2, remaining_cash=16)
        self.assertEqual(result, {"value": -7, "value_diff": -1, "size": -2, "size_diff": 0, "remaining_cash": 16, "reward": -1, "total_cash": 9})

    def test_calculate_step_hold_exist_short_position_with_negative_price_change(self):
        action = [1, -0.1]
        result = self.calc(action=action, close_price=3, position_value=-7, position_size=-2, remaining_cash=16)
        self.assertEqual(result, {"value": -6, "value_diff": 1, "size": -2, "size_diff": 0, "remaining_cash": 16, "reward": 1, "total_cash": 10})

    """
    Buy action
    """
    def test_calculate_step_buy_no_exist_position(self):
        amount = 0.77
        do_action = 0.1
        action = [amount, do_action]
        result = self.calc(action=action, close_price=0.01, position_value=0, position_size=0, remaining_cash=2.55)
        position = (2.55 * amount) / (1 + 0.0025)
        commission = rnd(position * 0.0025)
        size = rnd(position / 0.01)
        remained_cash = rnd(2.55 - position - commission)
        self.assertEqual(result, {"value": rnd(position), "value_diff": rnd(position), "size": size, "size_diff": size,
                                  "remaining_cash": remained_cash, "reward": -commission, "total_cash": rnd(position + remained_cash)})

    def test_calculate_step_buy_exist_long_position(self):
        amount = 0.5
        action = [amount, 0.2]
        new_close_price = 0.15
        result = self.calc(action=action, close_price=new_close_price, position_value=0.5, position_size=5, remaining_cash=1.25)
        total_amount = 1.25 + new_close_price * 5       # 2
        step_hold_reward = new_close_price * 5 - 0.5    # 0.25
        new_position_amount = total_amount * amount     # 1

        position_value_diff = (new_position_amount - new_close_price * 5) / (1 + 0.0025)    # 0.249377
        position_size_diff = rnd(position_value_diff / new_close_price)          # 1.662513

        commission = abs(rnd(position_value_diff * 0.0025))
        remained_cash = rnd(1.25 - abs(position_value_diff) - commission)
        total_step_reward = rnd(step_hold_reward - commission)
        self.assertEqual(result, {"value": rnd(new_position_amount - commission), "value_diff": rnd(position_value_diff),
                                  "size": 5 + position_size_diff, "size_diff": position_size_diff,
                                  "remaining_cash": remained_cash, "reward": total_step_reward,
                                  "total_cash": rnd(new_position_amount - commission + remained_cash)})

    def test_calculate_step_buy_exist_short_position(self):
        amount = 0.5
        action = [amount, 0.2]
        new_close_price = 0.15
        result = self.calc(action=action, close_price=new_close_price, position_value=-0.5, position_size=-5, remaining_cash=2.25)
        # hold result: {'value': -0.75, 'value_diff': -0.25, 'size': -5, 'size_diff': 0, 'remaining_cash': 2.25, 'reward': -0.25, 'total_cash': 1.5}

        total_cash = 2.25 - 5 * 0.15
        step_hold_reward = -0.25
        new_position_amount = total_cash * amount  # 0.75

        position_value_diff = (new_position_amount - new_close_price * -5) / (1 + 0.0025)  # 1.5 / comm = 1.4962
        position_size_diff = rnd(position_value_diff / new_close_price)  # 9.97

        commission = abs(position_value_diff * 0.0025)
        remained_cash = rnd(total_cash - new_close_price * -5 - position_value_diff - rnd(commission))
        total_step_reward = rnd(step_hold_reward - commission)
        self.assertEqual(result, {"value": rnd(new_position_amount - commission), "value_diff": rnd(position_value_diff),
                                  "size": rnd(-5 + position_size_diff), "size_diff": rnd(position_size_diff),
                                  "remaining_cash": rnd(remained_cash), "reward": rnd(total_step_reward),
                                  "total_cash": rnd(new_position_amount - commission + remained_cash)})

    """
    Sell action
    """
    def test_calculate_step_sell_no_exist_position_simple(self):
        amount = -0.5
        do_action = 0.1
        action = [amount, do_action]
        result = self.calc(action=action, close_price=0.01, position_value=0, position_size=0, remaining_cash=2)
        position = (2 * amount) / (1 + 0.0025)
        commission = abs(rnd(position * 0.0025))
        size = rnd(position / 0.01)
        remaining_cash = rnd(2 - position - commission)
        self.assertEqual(result, {"value": rnd(position), "value_diff": rnd(position),
                                  "size": size, "size_diff": size,
                                  "remaining_cash": remaining_cash, "reward": -commission,
                                  "total_cash": rnd(2 - commission)})

    def test_calculate_step_sell_no_exist_position(self):
        amount = -0.77
        do_action = 0.1
        action = [amount, do_action]
        result = self.calc(action=action, close_price=0.01, position_value=0, position_size=0, remaining_cash=2.55)
        position = (2.55 * amount) / (1 + 0.0025)               # -1.958603
        commission = abs(position * 0.0025)                # 0.004897
        size = rnd(position / 0.01)                             # -195.8603
        remaining_cash = rnd(2.55 - position - commission)
        self.assertEqual(result, {"value": rnd(position), "value_diff": rnd(position),
                                  "size": size, "size_diff": size,
                                  "remaining_cash": remaining_cash, "reward": -rnd(commission),
                                  "total_cash": rnd(2.55 - commission)})

    def test_calculate_step_sell_exist_long_position(self):
        amount = -0.5
        action = [amount, 0.2]
        new_close_price = 0.15
        result = self.calc(action=action, close_price=new_close_price, position_value=0.5, position_size=5, remaining_cash=1.25)
        total_amount = 1.25 + new_close_price * 5  # 2
        step_hold_reward = new_close_price * 5 - 0.5  # 0.25
        new_position_amount = total_amount * amount  # -1

        position_value_diff = (new_position_amount - new_close_price * 5) / (1 + 0.0025)  # -1.7456
        position_size_diff = rnd(position_value_diff / new_close_price)  # 1.662513

        commission = abs(rnd(position_value_diff * 0.0025))
        remaining_cash = rnd(total_amount - position_value_diff - 5 * new_close_price - commission)
        total_step_reward = rnd(step_hold_reward - commission)
        self.assertEqual(result, {"value": rnd(new_position_amount + commission), "value_diff": rnd(position_value_diff),
                                  "size": 5 + position_size_diff, "size_diff": position_size_diff,
                                  "remaining_cash": remaining_cash, "reward": total_step_reward,
                                  "total_cash": rnd(total_amount - commission)})

    def test_calculate_step_sell_exist_short_position(self):
        amount = -0.5
        action = [amount, 0.2]
        new_close_price = 0.15
        result = self.calc(action=action, close_price=new_close_price, position_value=-0.5, position_size=-5, remaining_cash=2.75)
        # {'value': -0.75, 'value_diff': -0.25, 'size': -5, 'size_diff': 0, 'remaining_cash': 2.25, 'reward': -0.25, 'total_cash': 2}
        total_amount = 2
        step_hold_reward = 0.15 * -5 + 0.5  # -0.25
        new_position_amount = total_amount * amount  # -1

        position_value_diff = (new_position_amount - new_close_price * -5) / (1 + 0.0025)  # -0.24
        position_size_diff = rnd(position_value_diff / new_close_price)  # 1.662513

        commission = abs(position_value_diff * 0.0025)
        remaining_cash = rnd(total_amount - position_value_diff + 5 * new_close_price - commission)
        total_step_reward = rnd(step_hold_reward - commission)
        self.assertEqual(result, {"value": rnd(new_position_amount + commission), "value_diff": rnd(position_value_diff),
                                  "size": -5 + position_size_diff, "size_diff": position_size_diff,
                                  "remaining_cash": remaining_cash, "reward": total_step_reward,
                                  "total_cash": rnd(total_amount - commission)})

    """
    Test flow: sell some -> price rise -> buy one -> buy more     
    """
    def test_flow_sell(self):
        amount = -0.2857142857      # sell 5 elements
        action = [amount, 1]
        result = self.calc(action=action, close_price=0.1, position_value=0, position_size=0, remaining_cash=1.75)
        value_wo_comission = -0.5
        value, size = value_wo_comission / (1 + 0.0025), (value_wo_comission / 0.1) / (1 + 0.0025)
        commission = abs(value_wo_comission - value)
        remaining_cash = 1.75 - value - commission
        total_cash = 1.75 - commission
        self.assertEqual(result, {"value": rnd(value), "value_diff": rnd(value), "size": rnd(size), "size_diff": rnd(size),
                                  "remaining_cash": rnd(remaining_cash), "reward": rnd(-commission),
                                  "total_cash": rnd(total_cash)})

    def test_flow_close(self):
        amount = 0.00071306      # close position (it's not null because of hack to include small commission)
        action = [amount, 1]
        result = self.calc(action=action, close_price=0.1, position_value=-0.498753, position_size=-4.987531, remaining_cash=2.247506)
        value, size = 0.0, 0.0
        value_diff, size_diff = 0.498753, 4.987531
        commission = 0.001247
        remaining_cash = 2.247506 - value_diff - commission
        total_cash = 1.748753 - commission
        self.assertEqual(result, {"value": rnd(value), "value_diff": rnd(value_diff), "size": rnd(size), "size_diff": rnd(size_diff),
                                  "remaining_cash": rnd(remaining_cash), "reward": rnd(-commission),
                                  "total_cash": rnd(total_cash)})
        self.assertEqual(result["remaining_cash"], result["total_cash"])


if __name__ == '__main__':
    unittest.main()
