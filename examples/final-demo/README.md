# final-demo

这个目录展示一个最小 Python 项目在本地被 review-pilot 扫出两个问题：

- 生产代码改动没有配套测试。
- 新增代码里留下 `print(` 调试输出。

推荐把这个目录复制到临时位置，再初始化 Git 仓库运行。

```bash
cp -R examples/final-demo /tmp/review-pilot-final-demo
cd /tmp/review-pilot-final-demo
git init
git add .
git commit -m 'initial demo project'
cp changed/calculator.py src/calculator.py
git add src/calculator.py
review-pilot review --staged --no-ai --format markdown --output review-report.md
```
