import runpy

if __name__ == "__main__":
    # for _ in range(10):
    runpy.run_module("Computational_analysis.VNS_QL", run_name="__main__")
    runpy.run_module("Computational_analysis.SA", run_name="__main__")
    runpy.run_module("Computational_analysis.VNS", run_name="__main__")