from TOTP import GenerateTOTPToken
from MainFile import TradingBot, LiveEngine

# --------------------------- MAIN ---------------------------

if __name__ == "__main__":
    flag = GenerateTOTPToken()

    if not flag:
        print("❌ TOTP generation failed")
        exit()

    print("TOTP generated")

    bot = TradingBot()
    app_id = bot.app_id
    raw_token = bot.access_token
    if not app_id or not raw_token:
        raise RuntimeError("app_id or token missing; check fyers_appid.txt / fyers_token.txt")

    # v3 websocket token format: "client_id:access_token". [web:50]
    ws_access_token = f"{app_id}:{raw_token}"
    engine = LiveEngine(bot, access_token=ws_access_token)
    engine.start()
    