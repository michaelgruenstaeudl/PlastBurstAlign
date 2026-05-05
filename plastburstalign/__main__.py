from .user_parameters import UserParametersScript
from .plastome_burst_and_align import PlastomeBurstAndAlign

def main():
    params = UserParametersScript()
    burst_align = PlastomeBurstAndAlign(params)
    burst_align.execute()
    print("\nend of script\n")

if __name__ == "__main__":
    main()

