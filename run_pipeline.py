import build_data_and_random
import calculate_2pcf
import run_rascalC
import fit_bao


def main():
    print("Step 1: build_data_and_random")
    build_data_and_random.main()

    print("Step 2: calculate_2pcf")
    calculate_2pcf.main()

    print("Step 3: run_rascalC")
    run_rascalC.main()

    print("Step 4: fit_bao")
    fit_bao.main()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()