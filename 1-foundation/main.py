from modules import adapter
import environ

env = environ.Env()
env.read_env()

def main():
    ad = adapter.Adapter(env)
    print(ad.chat("Hey, tell me a joke about computers."))

if __name__ == "__main__":
    main()
