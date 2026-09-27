import pytest

from twip.data import load_mat, thesis_path

FILTERING_TEST_12 = thesis_path("Tests", "Filtering Test", "filteringtest12.mat")
ALLAN_DATA = thesis_path("Tests", "Allan Variance", "allandata.mat")
IMPLEMENTATION_TEST_5 = thesis_path("Thesis", "Thesis Programs", "Implementation", "implementationtest5.txt")


def _require(path):
    if not path.exists():
        pytest.skip(f"thesis data file not found: {path}")
    return path


@pytest.fixture(scope="session")
def filtering_log():
    """Robot log from twip_v4 with raw IMU data and the on-board filter outputs."""
    return load_mat(_require(FILTERING_TEST_12))


@pytest.fixture(scope="session")
def allan_mat():
    import scipy.io as sio

    return sio.loadmat(str(_require(ALLAN_DATA)), squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="session")
def implementation_log():
    from twip.data import IMPLEMENTATION_COLUMNS, load_log

    return load_log(_require(IMPLEMENTATION_TEST_5), IMPLEMENTATION_COLUMNS)


@pytest.fixture(scope="session", params=[4, 5])
def v9_log(request):
    """Hardware LQR runs logged by the twip_v9 firmware."""
    from twip.data import IMPLEMENTATION_COLUMNS, load_log

    path = IMPLEMENTATION_TEST_5.with_name(f"implementationtest{request.param}.txt")
    return load_log(_require(path), IMPLEMENTATION_COLUMNS)
