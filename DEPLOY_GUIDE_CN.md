# GitHub + Streamlit Community Cloud deployment (简明步骤)

1. 将本目录 `models` 文件夹内放入你之前导出的 `mtgbm_365.json.gz`。
2. 访问 https://github.com/new，创建一个仓库，例如 `tips-outcome-navigator`；如果可以公开模型，选择 Public。注意：Public 仓库内的模型文件可以被下载。
3. 在 GitHub 仓库中选择 **Add file → Upload files**，将此目录**内部所有文件和文件夹**上传（不要上传整个 ZIP，也不要把外层文件夹多套一层）。确认根目录可直接看到 `app.py`、`requirements.txt`，且 `models/mtgbm_365.json.gz` 存在，然后提交（Commit）。不要上传原始患者数据。
4. 打开 https://share.streamlit.io，使用 GitHub 账号登录，点击 **Create app → Yup, I have an app**。
5. 选择刚才的仓库，Branch 设置 `main`，Main file path 设置 `app.py`。高级设置 Python 选择 **3.11**。
6. 点击 Deploy。部署成功后获得 `https://xxx.streamlit.app`，即使本机关闭也可以访问。长时间无人访问可能休眠，再次访问可唤醒。
7. 输入与你本地测试一致的8项指标，核对三项概率结果一致后再公开发布链接。

如出现 ModuleNotFoundError：核对对应 .py 与 requirements.txt 是否上传到仓库根目录。
如出现 model not connected：核对 GitHub 仓库中 `models/mtgbm_365.json.gz` 是否存在。
