import os

# Ignore the ported TAI tests by default; enable with HM_RUN_TAI_TESTS=1
collect_ignore_glob = []
if os.environ.get("HM_RUN_TAI_TESTS") != "1":
    collect_ignore_glob.append("tai/**")

