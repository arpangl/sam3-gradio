# SAM3 Gradio 服务器部署指南

## 如何从本地电脑访问服务器上的 Gradio 应用

### 方法一：使用 Gradio 公共链接（推荐）

1. 在服务器上运行应用：
   ```bash
   cd /home/jikangyi/sam3
   ./start_server.sh
   ```
   或者直接运行：
   ```bash
   python sam3_gradio_demo.py
   ```

2. 启动后，终端会显示一个公共链接（以 `https://xxxx.gradio.live` 格式），您可以在本地电脑的浏览器中打开这个链接访问应用。

### 方法二：通过服务器 IP 地址直接访问

1. 确保服务器防火墙允许 7860 端口的入站连接：
   ```bash
   # 如果使用 ufw
   sudo ufw allow 7860
   
   # 如果使用 iptables
   sudo iptables -A INPUT -p tcp --dport 7860 -j ACCEPT
   ```

2. 在本地电脑的浏览器中访问：
   ```
   http://服务器IP地址:7860
   ```
   例如：`http://123.45.67.89:7860`

### 方法三：使用 SSH 端口转发（安全方式）

1. 在本地电脑的终端中运行以下命令：
   ```bash
   ssh -L 7860:localhost:7860 用户名@服务器IP地址
   ```
   例如：`ssh -L 7860:localhost:7860 root@123.45.67.89`

2. 在本地电脑的浏览器中访问：
   ```
   http://localhost:7860
   ```

### 注意事项

- 确保服务器上有足够的 GPU 资源和内存来运行 SAM3 模型
- 处理大视频文件可能需要较长时间
- 如果使用公共链接，请注意数据隐私和安全问题
- 建议在生产环境中使用 HTTPS 和身份验证

### 故障排除

如果无法访问应用，请检查：

1. 服务器上的应用是否正在运行
2. 防火墙设置是否正确
3. 网络连接是否稳定
4. 服务器资源是否充足

### 后台运行

如果您希望应用在后台持续运行，可以使用 nohup 或 screen：

```bash
# 使用 nohup
nohup python sam3_gradio_demo.py > app.log 2>&1 &

# 使用 screen
screen -S sam3
python sam3_gradio_demo.py
# 按 Ctrl+A 然后 D 分离会话

# 重新连接到 screen 会话
screen -r sam3
```