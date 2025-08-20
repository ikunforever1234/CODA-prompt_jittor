import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Union

# ----------------------
# 1. 数据配置与常量定义
# ----------------------
DATA_CONSTRAINTS = {
    "y_min": 0.0,
    "y_max": 100.0,
    "k_min": 0.015, 
    "k_max": 0.1,
    "x0_min": 50,
    "x0_max": 300
}

# ----------------------
# 2. 基础配置
# ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------
# 3. 数据加载与预处理
# ----------------------
def load_employment_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.sort_values('ds').reset_index(drop=True)
        df = df.dropna(subset=['ds']).reset_index(drop=True)
        logging.info(f"数据加载完成，总样本量：{len(df)}（含空白值），时间范围：{df['ds'].min().date()}至{df['ds'].max().date()}")
        return df
    except Exception as e:
        logging.error(f"数据加载失败：{str(e)}")
        raise


# ----------------------
# 4. S型函数核心定义
# ----------------------
def logistic_func(x: Union[float, np.ndarray], L: float, k: float, x0: float) -> Union[float, np.ndarray]:
    return L / (1 + np.exp(-k * (x - x0)))


# ----------------------
# 5. 分年份拟合与补全
# ----------------------
def fit_yearly_s_curve(df: pd.DataFrame) -> pd.DataFrame:
    filled_df = df.copy()
    for year in filled_df['ds'].dt.year.unique():
        year_mask = filled_df['ds'].dt.year == year
        year_data = filled_df[year_mask].copy().sort_values('ds')
        year_data['day_of_year'] = year_data['ds'].dt.dayofyear
        x_all = year_data['day_of_year'].values
        y_obs = year_data['y'].values
        
        non_empty_mask = ~pd.isna(y_obs)
        x_obs = x_all[non_empty_mask]
        y_actual = y_obs[non_empty_mask]
        
        if len(x_obs) == 0:
            logging.warning(f"年份{year}无有效数据，无法拟合")
            continue
        
        y_actual = np.maximum.accumulate(y_actual)
        
        # 拟合参数初始化
        p0 = [
            min(DATA_CONSTRAINTS['y_max'], max(y_actual) * 1.1),
            0.02,
            x_obs[len(x_obs)//2]
        ]
        bounds = (
            [max(y_actual), DATA_CONSTRAINTS['k_min'], DATA_CONSTRAINTS['x0_min']],
            [DATA_CONSTRAINTS['y_max'], DATA_CONSTRAINTS['k_max'], DATA_CONSTRAINTS['x0_max']]
        )
        
        # 执行拟合
        try:
            popt, _ = curve_fit(
                logistic_func, x_obs, y_actual,
                p0=p0, bounds=bounds, maxfev=10000
            )
            L, k, x0 = popt
            logging.info(f"年份{year}拟合成功：L={L:.2f}, k={k:.4f}, x0={x0:.1f}")
        except Exception as e:
            logging.warning(f"年份{year}拟合失败，使用稳健参数：{str(e)}")
            L = min(DATA_CONSTRAINTS['y_max'], max(y_actual) * 1.1) if len(y_actual) > 0 else 100
            k = 0.02
            x0 = x_obs[len(x_obs)//2] if len(x_obs) > 0 else 180
        
        # 补全空白值
        prev_y = -np.inf
        for idx in year_data.index:
            x_current = year_data.loc[idx, 'day_of_year']
            if not pd.isna(year_data.loc[idx, 'y']):
                current_y = year_data.loc[idx, 'y']
                current_y = max(prev_y, current_y)
            else:
                current_y = logistic_func(x_current, L, k, x0)
                current_y = max(prev_y, min(DATA_CONSTRAINTS['y_max'], current_y))
            filled_df.loc[idx, 'y'] = round(current_y, 2)
            prev_y = current_y
    
    return filled_df


# ----------------------
# 6. 结果验证与可视化
# ----------------------
def validate_monotonicity(df: pd.DataFrame) -> Dict[int, bool]:
    validation = {}
    for year in df['ds'].dt.year.unique():
        year_data = df[df['ds'].dt.year == year]['y'].values
        is_increasing = np.all(np.diff(year_data) >= -1e-6)
        validation[year] = is_increasing
        status = "通过" if is_increasing else "未通过"
        logging.info(f"年份{year}单调性验证：{status}")
    return validation


def plot_yearly_trends(df: pd.DataFrame, output_path: str = "就业趋势图.png"):
    plt.figure(figsize=(12, 8))
    years = df['ds'].dt.year.unique()
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple']
    colors = [colors[i % len(colors)] for i in range(len(years))]
    
    for i, year in enumerate(years):
        year_data = df[df['ds'].dt.year == year].sort_values('ds')
        plt.plot(
            year_data['ds'].dt.dayofyear,
            year_data['y'],
            label=str(year),
            color=colors[i],
            marker='o', markersize=4
        )
    
    plt.title("分年份就业率趋势（S型拟合补全结果）")
    plt.xlabel("日序（1月1日为1）")
    plt.ylabel("就业率（%）")
    plt.ylim(DATA_CONSTRAINTS['y_min'], DATA_CONSTRAINTS['y_max'] + 5)
    plt.grid(alpha=0.3)
    plt.legend(title="年份")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"趋势图已保存至：{output_path}")
    plt.close()


# ----------------------
# 7. 主程序入口
# ----------------------
def main(input_file: str, output_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('拟合日志.log'), logging.StreamHandler()]
    )
    logging.info("===== S型函数就业率拟合程序启动 =====")
    
    try:
        df = load_employment_data(input_file)
        filled_df = fit_yearly_s_curve(df)
        validation = validate_monotonicity(filled_df)
        if not all(validation.values()):
            logging.warning("存在年份未通过单调性验证，建议检查原始数据")
        plot_yearly_trends(filled_df)
        filled_df.to_excel(output_file, index=False)
        logging.info(f"补全结果已保存至：{output_file}")
        
    except Exception as e:
        logging.error(f"程序执行失败：{str(e)}", exc_info=True)
    finally:
        logging.info("===== 程序执行结束 =====")


if __name__ == "__main__":
    main(
        input_file="标准化表格.xlsx",
        output_file="S型拟合补全结果.xlsx"
    )
