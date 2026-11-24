# pairwise-block-compress-mip

## 领域

游戏、图形与多媒体

## 主题

块压缩与 mip 链

## 说明

边界扩展、alpha 模式与层级依赖提交。

初始状态下 `graphics/block_compress_mip.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
