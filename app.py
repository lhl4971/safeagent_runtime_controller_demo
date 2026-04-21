import asyncio
from ui.layout import build_ui

auth_dict = {
    "ilyushine": "dF1vZ5sD5iM6",
    "liuhailin": "mOEYkUAwBuLn",
    "zhuming": "gVczW8LogFva",
    "nijie": "uTZaZRL8vjBX"
}


def verify_credentials(username: str, password: str) -> bool:
    return (username in auth_dict) and (password == auth_dict[username])


async def main():
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=22337,
        auth=verify_credentials,
        auth_message="VibeShell Agent Red Team Operation Platform Access Control",
        show_api=False
    )

if __name__ == "__main__":
    asyncio.run(main())
