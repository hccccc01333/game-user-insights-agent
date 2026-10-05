"""L2 特征字典 —— 特征的单一真源（single source of truth）。

设计要点：
    特征的名字、维度、类型、口径、缺失策略只在这里定义一次；
    extract_features.py 按本文件决定列顺序与取值，feature_dict.csv
    也从这里导出 —— 文档与实现不会漂移。

    （与 L2_features/README.md §3 一一对应；改口径 = 改这里 + 升
    FEATURE_VERSION，然后重跑。）
"""
from __future__ import annotations

from dataclasses import dataclass

FEATURE_VERSION = "l2v1"

# 兴趣品类向量：固定 top-N，词表从全量样本频次生成后冻结（见 README §10 A）
TAG_VECTOR_N = 20

# 资历分档边界（天），等距按年（见 README §10 B）
TENURE_BUCKET_BOUNDS = (365, 1095, 1825)


@dataclass(frozen=True)
class FeatureSpec:
    name: str      # 列名
    dim: str       # 维度：id / act / taste / rel / inf
    dtype: str     # int / float / bool / cat / list
    desc: str      # 口径说明
    source: str    # 来源数据面
    missing: str   # 缺失策略


BASE_FEATURES: list[FeatureSpec] = [
    # ── ① 身份 / 生命周期 id_ ──────────────────────────────
    FeatureSpec("uid_hash", "id", "str", "用户主键（加盐哈希）", "文件名", "必有"),
    FeatureSpec("feature_version", "id", "str", "特征口径版本", "本模块", "必有"),
    FeatureSpec("tenure_days", "id", "int", "注册天数（created_days，真实 tenure）", "detail.stat", "-1"),
    FeatureSpec("tenure_bucket", "id", "cat", "资历分档：<1y / 1-3y / 3-5y / >5y", "tenure_days", "unknown"),
    FeatureSpec("gender", "id", "cat", "性别（敏感）", "detail", "unknown"),
    FeatureSpec("country", "id", "cat", "国家/地区（敏感）", "detail", "unknown"),
    FeatureSpec("language", "id", "cat", "界面语言", "detail", "unknown"),
    FeatureSpec("ip_location", "id", "cat", "IP 属地（敏感，出产出域须转聚合）", "detail", "unknown"),
    FeatureSpec("is_silent", "id", "bool", "沉默账号标记", "detail", "False"),
    FeatureSpec("is_deactivated", "id", "bool", "注销标记", "detail", "False"),
    FeatureSpec("is_deleted", "id", "bool", "删除标记", "detail", "False"),
    FeatureSpec("account_status", "id", "cat", "账号状态：active/silent/deactivated/deleted", "detail", "unknown"),
    FeatureSpec("badge_count", "id", "int", "徽章总数", "detail.stat", "0"),
    FeatureSpec("badge_wear_count", "id", "int", "佩戴中的徽章数", "detail.wear_badges", "0"),
    FeatureSpec("first_seen_proxy_ts", "id", "float", "最早徽章时间（近似首次可见，非注册时间）", "badge", "null"),
    FeatureSpec("truncated_surfaces", "id", "str", "触达采集上限的数据面（分号分隔，质量标记）", "stat 总量 vs 采集条数", "空"),

    # ── ② 活跃 / 参与强度 act_ ─────────────────────────────
    FeatureSpec("review_count", "act", "int", "评价数", "detail.stat", "0"),
    FeatureSpec("moment_count", "act", "int", "动态数", "detail.stat", "0"),
    FeatureSpec("post_count", "act", "int", "帖子数", "detail.stat", "0"),
    FeatureSpec("topic_count", "act", "int", "话题数", "detail.stat", "0"),
    FeatureSpec("video_count", "act", "int", "视频数", "detail.stat", "0"),
    FeatureSpec("content_total", "act", "int", "内容产出总量（评价+动态+帖子+话题+视频）", "detail.stat", "0"),
    FeatureSpec("content_per_year", "act", "float", "年均内容产出（tenure<=0 时记 0）", "派生", "0"),
    FeatureSpec("played_app_count", "act", "int", "玩过（有游戏记录）的游戏数", "detail.stat", "0"),
    FeatureSpec("playing_app_count", "act", "int", "在玩的游戏数", "detail.stat", "0"),
    FeatureSpec("history_app_count", "act", "int", "历史浏览过的游戏数", "detail.stat", "0"),
    FeatureSpec("played_spent_total", "act", "int", "游戏总时长（分钟）", "detail.stat", "0"),
    FeatureSpec("reserved_count", "act", "int", "预约游戏数", "detail.stat", "0"),
    FeatureSpec("cloud_game_played_count", "act", "int", "云游戏游玩数", "detail.stat", "0"),
    FeatureSpec("recency_days", "act", "float", "距最近一次发布（评价/动态）的天数", "时间线", "null"),
    FeatureSpec("act_30d", "act", "int", "近 30 天评价+动态数", "时间线", "0"),
    FeatureSpec("act_90d", "act", "int", "近 90 天评价+动态数", "时间线", "0"),
    FeatureSpec("act_180d", "act", "int", "近 180 天评价+动态数", "时间线", "0"),
    FeatureSpec("decay_ratio", "act", "float", "近端活跃占比 act_30d / max(act_180d,1)", "派生", "0"),
    FeatureSpec("active_hour_top", "act", "cat", "发布时段众数（0-23，UTC+8）", "时间线", "null"),
    FeatureSpec("weekend_ratio", "act", "float", "周末事件占比（全部 exact 时间事件）", "时间线", "0"),
    FeatureSpec("device_top", "act", "cat", "发布设备众数（敏感）", "feed.device", "unknown"),

    # ── ③ 兴趣 / 游戏图谱 taste_ ───────────────────────────
    FeatureSpec("following_app_count", "taste", "int", "关注的游戏数", "detail.stat", "0"),
    FeatureSpec("favorite_app_count", "taste", "int", "收藏的游戏数", "detail.stat", "0"),
    FeatureSpec("wishlist_count", "taste", "int", "心愿单条数（口径 A：采集条数）", "wishlist", "0"),
    FeatureSpec("want_app_count", "taste", "int", "心愿单计数（口径 B：stat 计数）", "detail.stat", "0"),
    FeatureSpec("wishlist_locked", "taste", "bool", "心愿单被隐私设置锁定（非采集失败）", "show_setting/errors", "False"),
    FeatureSpec("distinct_tag_count", "taste", "int", "关注游戏覆盖的品类数", "following_app.tags", "0"),
    FeatureSpec("tag_entropy", "taste", "float", "品类分布香农熵（兴趣分散度）", "following_app.tags", "0"),
    FeatureSpec("genre_top1_ratio", "taste", "float", "最高频品类占比（兴趣集中度）", "following_app.tags", "0"),
    FeatureSpec("avg_follow_rating", "taste", "float", "关注游戏平均评分", "following_app.stat.rating", "null"),
    FeatureSpec("follow_rating_std", "taste", "float", "关注游戏评分标准差", "following_app.stat.rating", "null"),
    FeatureSpec("review_score_mean", "taste", "float", "用户打分的均值", "feed_review.review", "null"),
    FeatureSpec("review_score_std", "taste", "float", "用户打分的标准差", "feed_review.review", "null"),
    FeatureSpec("review_score_dist", "taste", "str", "打分分布（1:n;2:n;3:n;4:n;5:n）", "feed_review.review", "空"),
    FeatureSpec("dim_neg_rate_degree_of_freedom", "taste", "float", "自由度维度差评率", "feed_review.ratings", "null"),
    FeatureSpec("dim_neg_rate_gameplay", "taste", "float", "可玩性维度差评率", "feed_review.ratings", "null"),
    FeatureSpec("dim_neg_rate_operation", "taste", "float", "运营服务维度差评率", "feed_review.ratings", "null"),
    FeatureSpec("dim_neg_rate_visual_music", "taste", "float", "画面音乐维度差评率", "feed_review.ratings", "null"),
    FeatureSpec("wishlist_recent_ratio", "taste", "float", "心愿单近一年占比", "wishlist", "null"),

    # ── ④ 社交 / 关系 rel_ ────────────────────────────────
    FeatureSpec("following_count", "rel", "int", "关注用户数", "detail.stat", "0"),
    FeatureSpec("fans_count", "rel", "int", "粉丝数", "detail.stat", "0"),
    FeatureSpec("follower_ratio", "rel", "float", "粉丝/关注 比", "派生", "0"),
    FeatureSpec("verified_following_ratio", "rel", "float", "关注的官方/认证账号占比", "following_user.verified", "null"),
    FeatureSpec("verified_following_cov", "rel", "float", "verified 字段覆盖率（分母诚实声明）", "following_user", "null"),
    FeatureSpec("follow_source_cov", "rel", "float", "follow_source 字段覆盖率", "following_user", "null"),
    FeatureSpec("following_alive_ratio", "rel", "float", "关注对象中未注销占比", "following_user", "null"),
    FeatureSpec("fans_alive_ratio", "rel", "float", "粉丝中未注销占比", "fans", "null"),
    FeatureSpec("follow_source_top", "rel", "cat", "关注来源众数", "following_user", "unknown"),
    FeatureSpec("mutual_count", "rel", "int", "互关数（关注∩粉丝，仅计数不落标识）", "following_user + fans", "null"),
    FeatureSpec("following_hashtag_count", "rel", "int", "关注话题数", "detail.stat", "0"),
    FeatureSpec("following_developer_count", "rel", "int", "关注厂商数", "detail.stat", "0"),
    FeatureSpec("forum_count", "rel", "int", "加入论坛数", "detail.stat", "0"),

    # ── ⑤ 内容影响力 / 口碑 inf_ ──────────────────────────
    FeatureSpec("voteup_received", "inf", "int", "累计获赞数", "detail.stat", "0"),
    FeatureSpec("votefunny_received", "inf", "int", "累计获「欢乐」数", "detail.stat", "0"),
    FeatureSpec("be_voted_up_review", "inf", "int", "评价被点赞数", "detail.stat", "0"),
    FeatureSpec("be_voted_up_moment", "inf", "int", "动态被点赞数", "detail.stat", "0"),
    FeatureSpec("be_favorited_count", "inf", "int", "内容被收藏数", "detail.stat", "0"),
    FeatureSpec("avg_ups_per_moment", "inf", "float", "评价/动态平均点赞", "feed.stat.ups", "null"),
    FeatureSpec("max_ups", "inf", "int", "单条最高点赞", "feed.stat.ups", "0"),
    FeatureSpec("interaction_speed_median", "inf", "float", "发布到被评论中位间隔（秒）", "feed", "null"),
    FeatureSpec("favorite_moment_count", "inf", "int", "收藏的动态数", "detail.stat", "0"),
    FeatureSpec("purchased_app_count", "inf", "int", "购买过的游戏数", "detail.stat", "0"),
    FeatureSpec("app_achievement_count", "inf", "int", "游戏成就数", "detail.stat", "0"),
]

# 兴趣品类向量列（固定 top-N，词表运行时冻结）
TAG_FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        f"tag_top{i:02d}", "taste", "float",
        f"关注游戏品类词表第 {i} 位占比", "following_app.tags", "0",
    )
    for i in range(1, TAG_VECTOR_N + 1)
]

FEATURES: list[FeatureSpec] = BASE_FEATURES + TAG_FEATURES


def column_names() -> list[str]:
    """特征表的列顺序（uid_hash / feature_version 在最前）。"""
    return [f.name for f in FEATURES]


def export_rows() -> list[dict]:
    """导出为可落 CSV 的行（供 extract_features 写 feature_dict.csv）。"""
    return [
        {
            "name": f.name,
            "dim": f.dim,
            "dtype": f.dtype,
            "desc": f.desc,
            "source": f.source,
            "missing_policy": f.missing,
        }
        for f in FEATURES
    ]