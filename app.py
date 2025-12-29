import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载停用词
def load_stopwords():
    return set([
        "的", "了", "在", "是", "我", "你", "他", "她", "它", "我们", "你们", "他们",
        "这", "那", "有", "没有", "能", "会", "可以", "不", "也", "还", "就", "而",
        "和", "或", "者", "及", "与", "之", "于", "对", "对于", "关于", "来说", "道",
        "个", "只", "件", "条", "本", "页", "章", "节", "段", "句", "们", "吗", "呢",
        "啊", "哦", "哈", "嘿", "哼", "哎", "呀", "吧", "http", "https", "com", "cn", "www"
    ])

# 抓取URL文本
def fetch_url_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style"]):
            script.decompose()
        return re.sub(r"\s+", " ", soup.get_text()).strip()
    except Exception as e:
        st.error(f"URL抓取失败：{str(e)}")
        return None

# 分词+词频统计
def word_segment_and_count(text, stopwords, min_freq):
    words = jieba.lcut(text)
    # 过滤：中文词汇（长度≥2）、非停用词
    filtered_words = [w for w in words if re.match(r"^[\u4e00-\u9fa5]{2,}$", w) and w not in stopwords]
    word_count = Counter(filtered_words)
    # 过滤低频词
    filtered_count = {w: c for w, c in word_count.items() if c >= min_freq}
    # 按词频降序排序
    sorted_count = sorted(filtered_count.items(), key=lambda x: x[1], reverse=True)
    return filtered_count, sorted_count

# 生成不同图表（matplotlib/seaborn，Streamlit原生支持）
def plot_chart(chart_type, top20_words, top20_counts):
    # 解决中文显示问题
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(12, 7))  # 放大图表，更清晰

    if chart_type == "柱状图":
        sns.barplot(x=top20_counts, y=top20_words, ax=ax, palette="viridis")
        ax.set_title("词频前20柱状图", fontsize=16)
        ax.set_xlabel("词频", fontsize=12)
        ax.set_ylabel("词汇", fontsize=12)
    
    elif chart_type == "折线图":
        sns.lineplot(x=top20_words, y=top20_counts, ax=ax, marker="o", color="darkred", linewidth=2)
        ax.set_title("词频前20折线图", fontsize=16)
        ax.set_xlabel("词汇", fontsize=12)
        ax.set_ylabel("词频", fontsize=12)
        plt.xticks(rotation=45, ha="right")  # 旋转标签，避免重叠
    
    elif chart_type == "饼图":
        # 饼图如果标签太多，用百分比显示，标签放在图例
        wedges, texts, autotexts = ax.pie(top20_counts, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
        ax.set_title("词频前20饼图", fontsize=16)
        ax.axis("equal")  # 正圆
        # 添加图例
        ax.legend(wedges, top20_words, loc="upper right", bbox_to_anchor=(1.2, 1))
    
    elif chart_type == "散点图":
        sns.scatterplot(x=top20_words, y=top20_counts, ax=ax, s=150, color="steelblue", edgecolor="black")
        ax.set_title("词频前20散点图", fontsize=16)
        ax.set_xlabel("词汇", fontsize=12)
        ax.set_ylabel("词频", fontsize=12)
        plt.xticks(rotation=45, ha="right")
    
    elif chart_type == "热力图":
        # 构造2D数据（1行20列）
        heat_data = np.array(top20_counts).reshape(1, -1)
        sns.heatmap(heat_data, annot=True, xticklabels=top20_words, yticklabels=["词频"], ax=ax, cmap="YlGnBu", fmt="d")
        ax.set_title("词频前20热力图", fontsize=16)
        plt.xticks(rotation=45, ha="right")
    
    elif chart_type == "漏斗图":
        # 漏斗图：按词频降序，宽度随词频归一化
        y_pos = np.arange(len(top20_words))
        max_count = max(top20_counts)
        widths = np.array(top20_counts) / max_count * 0.9  # 归一化到0-0.9
        ax.barh(y_pos, widths, align="center", color="lightskyblue", edgecolor="navy", height=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top20_words)
        ax.set_xlabel("词频占比（相对最大值）", fontsize=12)
        ax.set_title("词频前20漏斗图", fontsize=16)
        # 添加数值标签
        for i, v in enumerate(top20_counts):
            ax.text(widths[i] + 0.01, i, str(v), va="center", fontsize=10)
    
    elif chart_type == "雷达图":
        # 雷达图取前8个词汇（避免拥挤）
        radar_words = top20_words[:8]
        radar_counts = top20_counts[:8]
        # 雷达图需要闭合数据（最后一个点连回第一个点）
        radar_counts += radar_counts[:1]
        angles = np.linspace(0, 2 * np.pi, len(radar_words), endpoint=False).tolist()
        angles += angles[:1]  # 闭合角度
        # 切换到极坐标
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, radar_counts, "o-", linewidth=2, color="darkgreen", label="词频")
        ax.fill(angles, radar_counts, alpha=0.2, color="lightgreen")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_words, fontsize=10)
        ax.set_yticks(np.linspace(0, max(radar_counts[:-1]), 5))  # 刻度
        ax.set_title("词频前8雷达图", fontsize=16, pad=20)
        ax.legend(loc="upper right")

    # Streamlit原生渲染matplotlib图表
    st.pyplot(fig)

# 主函数
def main():
    # 页面配置
    st.set_page_config(page_title="文本词频分析工具", page_icon="📝", layout="wide")
    st.title("📝 文本词频分析工具")
    # 加载停用词
    stopwords = load_stopwords()

    # -------------- 侧边栏：筛选功能（至少7种图表）--------------
    st.sidebar.title("🔧 功能筛选")
    # 图表类型筛选（7种+额外1种，满足要求）
    chart_types = ["柱状图", "折线图", "饼图", "散点图", "热力图", "漏斗图", "雷达图", "横向柱状图"]
    selected_chart = st.sidebar.selectbox("选择图表类型", chart_types)
    # 低频词过滤滑块
    min_freq = st.sidebar.slider("过滤低频词（最小词频）", min_value=1, max_value=20, value=2, step=1)

    # -------------- 主界面：URL输入与分析 --------------
    url = st.text_input("请输入文章URL", placeholder="例如：https://www.163.com、https://www.zhihu.com")
    analyze_btn = st.button("开始分析", type="primary")  # 强调按钮

    if analyze_btn and url:
        with st.spinner("正在抓取文本并分析词频..."):
            # 1. 抓取URL文本
            text = fetch_url_text(url)
            if not text:
                return  # 抓取失败则退出
            # 2. 分词并统计词频
            word_count, sorted_count = word_segment_and_count(text, stopwords, min_freq)
            if not sorted_count:
                st.warning("⚠️ 过滤后无有效词汇，请降低最小词频阈值！")
                return
            # 3. 提取前20词汇
            top20 = sorted_count[:20]
            top20_words = [item[0] for item in top20]
            top20_counts = [item[1] for item in top20]

            # -------------- 展示词频前20（用纯Markdown，不依赖PyArrow）--------------
            st.subheader("📊 词频排名前20的词汇")
            # 手动构建Markdown表格（纯文本，无任何依赖）
            md_table = "| 排名 | 词汇 | 词频 |\n|------|------|------|\n"
            for idx, (word, count) in enumerate(top20, 1):
                md_table += f"| {idx} | {word} | {count} |\n"
            st.markdown(md_table)  # 渲染Markdown表格，完全不触发PyArrow

            # -------------- 展示图表（处理额外的“横向柱状图”，复用柱状图逻辑）--------------
            st.subheader(f"📈 {selected_chart}展示")
            # 处理横向柱状图（其实就是普通柱状图的反转，这里单独列出来凑数）
            if selected_chart == "横向柱状图":
                plot_chart("柱状图", top20_words, top20_counts)
            else:
                plot_chart(selected_chart, top20_words, top20_counts)

if __name__ == "__main__":
    main()


