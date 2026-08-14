"""
仓库共现关联图谱生成脚本

数据来源: GitHub GraphQL API 的 contributionsCollection
逻辑:
  1. 拉取过去 N 天内,每个仓库的逐日提交次数
  2. 按自然周聚合,只要两个仓库在同一周都有提交,视为一次"共现"
  3. 共现次数 = 边权重;仓库总提交数 = 节点大小
  4. 用 networkx 的力导向布局(spring_layout)计算坐标
  5. 用 matplotlib 渲染成暗色终端风格的 SVG,输出到 OUTPUT_PATH

环境变量:
  GITHUB_TOKEN   - 具有 read:user 权限的 Personal Access Token(必需)
  GITHUB_LOGIN   - 要分析的 GitHub 用户名(必需)
  DAYS_BACK      - 回溯天数,默认 365
  OUTPUT_PATH    - 输出 SVG 路径,默认 contribution-graph.svg
"""

import os
import sys
import datetime
import itertools
import collections

import requests
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GITHUB_LOGIN = os.environ.get("GITHUB_LOGIN")
DAYS_BACK = int(os.environ.get("DAYS_BACK", "365"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "contribution-graph.svg")

GRAPHQL_URL = "https://api.github.com/graphql"

# GitHub 的 contributionsCollection 单次查询窗口不能超过 1 年,
# 超过 365 天需要分段查询后再合并
QUERY = """
query($login: String!, $from: DateTime!, $to: DateTime!) {
  user(login: $login) {
    contributionsCollection(from: $from, to: $to) {
      commitContributionsByRepository(maxRepositories: 100) {
        repository {
          name
          primaryLanguage { name }
        }
        contributions(first: 100) {
          nodes {
            occurredAt
            commitCount
          }
          pageInfo {
            hasNextPage
            endCursor
          }
        }
      }
    }
  }
}
"""


def fetch_window(login: str, start: datetime.datetime, end: datetime.datetime) -> dict:
    if not GITHUB_TOKEN:
        raise RuntimeError("缺少环境变量 GITHUB_TOKEN")
    headers = {"Authorization": f"bearer {GITHUB_TOKEN}"}
    variables = {
        "login": login,
        "from": start.isoformat() + "Z",
        "to": end.isoformat() + "Z",
    }
    resp = requests.post(
        GRAPHQL_URL,
        json={"query": QUERY, "variables": variables},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL 返回错误: {payload['errors']}")
    return payload["data"]["user"]["contributionsCollection"]


def collect_repo_weekly_activity(login: str, days_back: int) -> dict:
    """
    返回 { repo_name: set(iso_week_key, ...) }
    iso_week_key 形如 "2026-W07",用于判断"同一周是否都有提交"
    """
    now = datetime.datetime.utcnow()
    repo_weeks: dict[str, set] = collections.defaultdict(set)
    repo_lang: dict[str, str] = {}
    repo_total_commits: dict[str, int] = collections.defaultdict(int)

    # 分段拉取,每段最长 365 天(GitHub API 限制)
    window_end = now
    remaining = days_back
    while remaining > 0:
        span = min(remaining, 365)
        window_start = window_end - datetime.timedelta(days=span)
        data = fetch_window(login, window_start, window_end)

        for entry in data["commitContributionsByRepository"]:
            repo_name = entry["repository"]["name"]
            lang = entry["repository"]["primaryLanguage"]
            repo_lang.setdefault(repo_name, lang["name"] if lang else "Unknown")

            for node in entry["contributions"]["nodes"]:
                occurred = datetime.datetime.fromisoformat(
                    node["occurredAt"].replace("Z", "+00:00")
                )
                iso_year, iso_week, _ = occurred.isocalendar()
                week_key = f"{iso_year}-W{iso_week:02d}"
                repo_weeks[repo_name].add(week_key)
                repo_total_commits[repo_name] += node["commitCount"]

        window_end = window_start
        remaining -= span

    return repo_weeks, repo_lang, repo_total_commits


def build_cooccurrence_graph(repo_weeks: dict, repo_lang: dict, repo_total_commits: dict) -> nx.Graph:
    graph = nx.Graph()

    for repo, total in repo_total_commits.items():
        if total == 0:
            continue
        graph.add_node(repo, size=total, lang=repo_lang.get(repo, "Unknown"))

    # 两两仓库比较共同周数
    repos = [r for r in repo_weeks if repo_total_commits.get(r, 0) > 0]
    for repo_a, repo_b in itertools.combinations(repos, 2):
        shared = repo_weeks[repo_a] & repo_weeks[repo_b]
        if shared:
            graph.add_edge(repo_a, repo_b, weight=len(shared))

    return graph


def render_svg(graph: nx.Graph, output_path: str) -> None:
    if graph.number_of_nodes() == 0:
        print("没有可渲染的节点,跳过绘图", file=sys.stderr)
        return

    # 暗色终端风格配色
    bg_color = "#0d1117"
    node_color = "#39d353"  # GitHub 贡献墙同款绿色
    edge_color = "#2f81f7"
    text_color = "#c9d1d9"

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
    ax.set_facecolor(bg_color)

    pos = nx.spring_layout(graph, k=0.9, weight="weight", seed=42)

    sizes = [graph.nodes[n]["size"] for n in graph.nodes]
    max_size = max(sizes) if sizes else 1
    node_sizes = [200 + 1800 * (s / max_size) for s in sizes]

    weights = [graph.edges[e]["weight"] for e in graph.edges]
    max_weight = max(weights) if weights else 1
    edge_widths = [0.5 + 3 * (w / max_weight) for w in weights]

    nx.draw_networkx_edges(
        graph, pos, ax=ax, width=edge_widths, edge_color=edge_color, alpha=0.35
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=node_sizes, node_color=node_color, alpha=0.85,
        linewidths=0,
    )
    nx.draw_networkx_labels(
        graph, pos, ax=ax, font_size=9, font_color=text_color,
        font_family="monospace", verticalalignment="top",
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, format="svg", facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成图谱: {output_path}  节点数={graph.number_of_nodes()}  边数={graph.number_of_edges()}")


def main() -> None:
    if not GITHUB_LOGIN:
        raise RuntimeError("缺少环境变量 GITHUB_LOGIN")

    repo_weeks, repo_lang, repo_total_commits = collect_repo_weekly_activity(
        GITHUB_LOGIN, DAYS_BACK
    )
    graph = build_cooccurrence_graph(repo_weeks, repo_lang, repo_total_commits)
    render_svg(graph, OUTPUT_PATH)


if __name__ == "__main__":
    main()
