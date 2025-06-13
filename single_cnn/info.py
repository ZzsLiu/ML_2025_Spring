from datetime import datetime


def info_out(message, thread_id=None):
    """带线程ID的日志输出"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]"

    if thread_id is not None:
        log_entry += f" [线程{thread_id}]"

    log_entry += f" {message}"

    #print(log_entry)
    # with open(f"logs/training_log_{thread_id or 'main'}.txt", "a", encoding="utf-8") as f:
    #     f.write(log_entry + "\n")
    with open(f"../logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")