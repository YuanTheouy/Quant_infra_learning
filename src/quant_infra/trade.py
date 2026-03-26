# -*- coding: utf-8 -*-
"""
交易模拟模块：更真实地模拟小资金交易
- 增加手续费 + 滑点（单边 1%，来回共 2%）
- 过滤创业板 / 科创板 / 北交所 及高价股（收盘价 > 100 元）
- 导出每个换仓时间点的持仓记录 → output/trade_holdings.csv
- 导出日频净值（兼容 group_plot，line='long'）→ {factor_table}_trade_daily_ret
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from quant_infra import db_utils
from quant_infra.const import *
from quant_infra.get_data import get_ins, get_index_data
from quant_infra.factor_calc import winsorize
from quant_infra.factor_analyze import group_plot


def simulate_trade(
    factor_table,
    fac_freq,
    bench_index="000002.SH",
    other_name=None,
    sample="全市场",
    n_top=5,
    factor_direction="大",
    commission_rate=0.001,
    slippage_rate=0.01,
    price_max=100.0,
    filter_boards=("gem", "star", "bj"),
):
    """
    真实交易模拟回测

    Args:
        factor_table(str): 因子表名（DuckDB 中的表名，如 'f1'）
        fac_freq(str): 因子频率（如 '月度'）
        bench_index(str): 基准指数代码（默认 '000002.SH'）
        other_name(str): 因子列原始名称（默认 None，提供则重命名为 'factor'）
        sample(str): 回测样本，如 '全市场'、'中证1000'，默认 '全市场'
        n_top(int): 每期买入股票数（默认 5）
        factor_direction(str): 因子方向，'大' 表示因子越大越好（取前 n_top），'小' 表示因子越小越好（取后 n_top）
        commission_rate(float): 单边手续费率（默认 0.001 即 0.1%）
        slippage_rate(float): 单边滑点率（默认 0.01 即 1%，来回共 2%）
        price_max(float): 股价过滤上限（默认 100 元）
        filter_boards(tuple/list/set): 交易权限过滤，指定要排除的板块。
            可选值：'gem'（创业板，300开头）、'star'（科创板，68开头）、'bj'（北交所，4/8开头）。
            默认 ('gem', 'star', 'bj') 过滤全部三个板块，传入空元组则不过滤。
    """
    # ----- 1. 加载因子数据 -----
    try:
        fac = db_utils.read_sql(f"SELECT * FROM {factor_table}")
        if other_name:
            fac.rename(columns={other_name: "factor"}, inplace=True)
        fac["date"] = pd.to_datetime(fac["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    except Exception as e:
        print(f"错误：无法读取因子表 {factor_table}: {e}")
        return None

    # ----- 2. 加载股票日行情（含 close 用于价格过滤） -----
    stk = db_utils.read_sql("SELECT ts_code, trade_date, pct_chg, close FROM stock_bar")
    stk["date"] = pd.to_datetime(stk["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    stk["ret"] = stk["pct_chg"] / 100.0
    stk["ret"] = winsorize(stk["ret"], n=3)

    # 股票名称字典：ts_code → name
    name_df = db_utils.read_sql("SELECT ts_code, name FROM stock_basic")
    code_to_name: dict = dict(zip(name_df["ts_code"], name_df["name"]))

    # ----- 3. 过滤不可购买板块 -----
    _board_prefixes = {"gem": "300", "star": "68", "bj": ("4", "8")}
    for board in filter_boards:
        prefix = _board_prefixes.get(board)
        if prefix:
            fac = fac[~fac["ts_code"].str.startswith(prefix)]
            stk = stk[~stk["ts_code"].str.startswith(prefix)]

    # ----- 4. 生成周期列 -----
    p_freq = FREQ_MAP.get(fac_freq)
    if not p_freq:
        raise ValueError(f"不支持的频率: {fac_freq}")
    period_col = f"date_{p_freq}"

    fac[period_col] = fac["date"].dt.to_period(p_freq)
    stk[period_col] = stk["date"].dt.to_period(p_freq)

    # ----- 5. 基准收益 -----
    bench_raw = get_index_data(bench_index)
    bench_raw["bench_ret"] = bench_raw["pct_chg"] / 100.0
    bench_df = bench_raw[["trade_date", "bench_ret"]].copy()

    # ----- 6. 样本范围 -----
    if sample == "全市场":
        sample_fac = fac.copy()
        sample_stk = stk.copy()
    else:
        ## 保留样本内的股票
        index_code = INDEX_NAME_TO_CODE[sample]
        ins_path = Path(f"./Data/Metadata/{index_code}_ins.csv")
        constituent_stocks = (
            set(pd.read_csv(ins_path)["con_code"].unique())
            if ins_path.exists()
            else get_ins(index_code)
        )
        sample_fac = fac[fac["ts_code"].isin(constituent_stocks)].copy()
        sample_stk = stk[stk["ts_code"].isin(constituent_stocks)].copy()

    # 价格快照：供换仓日过滤高价股
    price_snapshot = sample_stk[["ts_code", "trade_date", "close"]].drop_duplicates()

    all_holdings = []

    # ----- 7. 计算每期换仓持仓 -----
    fac_last = (
        sample_fac.dropna(subset=["factor"])
        .sort_values(["ts_code", "trade_date"])
        .groupby(["ts_code", period_col], as_index=False)
        .agg(
            ## 因子值取均值后除以标准差，以达到平滑因子值，减少极端值对换仓的影响
            # factor=("factor", lambda x: x.mean() / x.std() if len(x) >= 2 and x.std() > 0 else np.nan),
            factor=("factor", "last"),
            trade_date=("trade_date", "max"),
        )
    )

    fac_with_price = fac_last.merge(
        price_snapshot[["ts_code", "trade_date", "close"]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    fac_with_price = fac_with_price[fac_with_price["close"] <= price_max]

    holdings = {}  # hold_period (Period) → list[ts_code]
    for period, group in fac_with_price.groupby(period_col):
        # True for '小'，False for '大'
        is_ascending = factor_direction == "小" 
        ranked = group.sort_values("factor", ascending=is_ascending)
        if ranked.empty:
            continue
        selected = ranked.head(n_top)["ts_code"].tolist()

        hold_period = period + 1
        holdings[hold_period] = selected

        rebal_date = int(group["trade_date"].max())
        all_holdings.append(
            {
                "样本": sample,
                "换仓日": rebal_date,
                "持有周期": str(hold_period),
                "持仓股票": ",".join(selected),
                "持仓股票名称": ",".join(code_to_name.get(c, c) for c in selected),
                "持仓数量": len(selected),
            }
        )

    # ----- 8. 计算日度组合收益（含交易成本） -----
    daily_ret = _compute_portfolio_daily_ret(
        sample_stk, holdings, period_col, commission_rate, slippage_rate
    )
    if daily_ret.empty:
        print("无有效回测结果")
        return

    daily_ret = daily_ret.merge(bench_df, on="trade_date", how="inner")
    daily_ret["样本"] = sample
    daily_ret["频率"] = fac_freq

    long_ret = daily_ret["long"].dropna()
    SR = (
        float(long_ret.mean() / long_ret.std() * np.sqrt(242))
        if long_ret.std() > 0
        else np.nan
    )
    print(f"[{sample} - {fac_freq}] 多头年化夏普（含交易成本）: {round(SR, 2)}")

    holdings_df = pd.DataFrame(all_holdings).sort_values("换仓日", ascending = False, ignore_index=True)

    # ----- 9. 写出结果 -----
    db_utils.write_to_db(daily_ret, f"{factor_table}_trade_daily_ret", save_mode="replace")

    output_path = Path("factor_mining") / factor_table / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    holdings_df.to_csv(output_path / "trade_holdings.csv", index=False, encoding="utf-8-sig")

    print(f"交易模拟完成，持仓记录已保存到 {output_path}/trade_holdings.csv")

    # ----- 10. 绘图 -----
    group_plot(sample=sample, freq=fac_freq, line="long", factor_table=factor_table, mode="trade")


def _compute_portfolio_daily_ret(stk_df, holdings, period_col, commission_rate, slippage_rate):
    """
    按换仓周期合并日度等权收益，在每期首日扣除双边交易成本。

    Args:
        stk_df: 含 [ts_code, trade_date, ret, period_col] 的日行情 DataFrame
        holdings: {hold_period (Period): list[ts_code]}
        period_col: 周期列名
        commission_rate: 单边手续费率
        slippage_rate: 单边滑点率

    Returns:
        DataFrame[trade_date, long]（trade_date 为 int）
    """
    if not holdings:
        return pd.DataFrame()

    sorted_periods = sorted(holdings.keys())
    prev_set: set = set()
    result_rows = []

    for hold_period in sorted_periods:
        curr_set = set(holdings[hold_period])
        if not curr_set:
            prev_set = curr_set
            continue

        # 换仓成本：买入 / 卖出各含一次滑点和手续费
        if not prev_set:
            # 首次建仓：仅买入成本
            cost = 1.0 * (slippage_rate + commission_rate)
        else:
            n_prev = len(prev_set)
            n_curr = len(curr_set)
            sold = prev_set - curr_set
            bought = curr_set - prev_set
            sell_frac = len(sold) / n_prev if n_prev > 0 else 0.0
            buy_frac = len(bought) / n_curr if n_curr > 0 else 0.0
            cost = (sell_frac + buy_frac) * (slippage_rate + commission_rate)

        # 当期持仓股票的日收益
        period_stk = stk_df[
            (stk_df[period_col] == hold_period) & stk_df["ts_code"].isin(curr_set)
        ]
        if period_stk.empty:
            prev_set = curr_set
            continue

        daily_eq = (
            period_stk.groupby("trade_date", as_index=False)
            .agg(long=("ret", "mean"))
            .sort_values("trade_date")
            .reset_index(drop=True)
        )

        # 首日扣除换仓成本
        daily_eq.loc[0, "long"] -= cost

        result_rows.append(daily_eq)
        prev_set = curr_set

    if not result_rows:
        return pd.DataFrame()

    return pd.concat(result_rows, ignore_index=True).sort_values("trade_date").reset_index(drop=True)


