# Notebook Export Helper

如果 `ipynb` 导出的 HTML 文件太大，DeepSeek 无法完整读取，请使用本仓库中的导出脚本将 Notebook 导出为更轻量的格式。

## 功能

- 生成去掉输出的 `*.ipynb` 副本
- 导出为 `html`、`markdown` 或 `script`
- 默认会去掉代码输出，减少 `html` 大小

## 使用方法

1. 进入仓库目录：

```bash
cd "D:\mrGuo\我爱的美在这里\Self-Improvement\class\CS50\deeplearning\Vehicle-Classification-Using-Machine-Learning-and-Deep-Learning-main"
```

2. 导出单个 notebook：

```bash
python export_notebooks_clean.py "VEHICLE CLASSIFICATION_FIXED (1).ipynb" --formats html markdown script --output-dir exported_notebooks
```

3. 导出当前目录下所有 notebook：

```bash
python export_notebooks_clean.py . --formats html markdown script --output-dir exported_notebooks
```

4. 只生成去掉输出的 notebook（不导出 HTML）：

```bash
python export_notebooks_clean.py "VEHICLE CLASSIFICATION_FIXED (1).ipynb" --clean-only --output-dir exported_notebooks
```
4. 仅导出 notebook 输出，隐藏代码输入：

```bash
python export_notebooks_clean.py "VEHICLE CLASSIFICATION_FIXED (1).ipynb" --output-only --formats html markdown --output-dir exported_notebooks
```
## 说明

- `html` 导出会默认使用 `--TemplateExporter.exclude_output=True`，这样会省去输出结果和图片数据
- `markdown` 导出也默认去掉输出，使文件更小、更易搜索
- 如果你只需要代码，`script` 导出会生成一个普通的 Python 脚本

## 输出路径

默认导出位置为：

```
exported_notebooks/
```

如果需要修改输出目录，可使用 `--output-dir` 参数。
