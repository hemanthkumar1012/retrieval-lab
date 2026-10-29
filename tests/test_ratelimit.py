from app.ratelimit import allow


class FakeRedis:
    def __init__(self):
        self.counts, self.ttls = {}, {}

    def incr(self, k):
        self.counts[k] = self.counts.get(k, 0) + 1
        return self.counts[k]

    def expire(self, k, seconds):
        self.ttls[k] = seconds


class DownRedis:
    def incr(self, k):
        raise ConnectionError("redis is down")


def test_blocks_after_limit():
    r = FakeRedis()
    results = [allow(r, "ip1", 3, now=1000) for _ in range(5)]
    assert results == [True, True, True, False, False]


def test_keys_are_independent():
    r = FakeRedis()
    assert allow(r, "a", 1, now=1000)
    assert not allow(r, "a", 1, now=1000)
    assert allow(r, "b", 1, now=1000)


def test_window_rolls_over():
    r = FakeRedis()
    assert allow(r, "a", 1, window=60, now=1000)
    assert not allow(r, "a", 1, window=60, now=1010)
    assert allow(r, "a", 1, window=60, now=1000 + 60)


def test_expiry_is_set_once_per_bucket():
    r = FakeRedis()
    allow(r, "a", 5, window=60, now=1000)
    allow(r, "a", 5, window=60, now=1001)
    assert list(r.ttls.values()) == [120]


def test_fails_open_when_redis_is_down():
    assert allow(DownRedis(), "a", 1, now=1000) is True
