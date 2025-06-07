import swim_model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="命令行参数示例")

    # 添加参数
    parser.add_argument("--experiment", type=str, required=True, help="输入文件路径")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--cuda", action="store_true", help="是否使用CUDA")

    args = parser.parse_args()