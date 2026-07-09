# final-demo

这个目录展示一个最小 Python 项目在本地被 review-pilot 扫出两个问题：

- 生产代码改动没有配套测试。
- 新增代码里留下 `print(` 调试输出。

把这个目录复制到临时目录，再初始化 Git 仓库运行：

```bash
cp -R examples/final-demo /tmp/review-pilot-final-demo
cd /tmp/review-pilot-final-demo
git init
git add .
git commit -m 'initial demo project'
cp changed/demo_app/calculator.py demo_app/calculator.py
git add demo_app/calculator.py
review-pilot review --staged --no-ai --format markdown --output review-report.md
```
