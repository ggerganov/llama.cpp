#include "../src/llama-moe-cache-policy.h"

#include <cstdio>

static bool check_replay() {
    llama_moe_cache_policy p;
    p.init(2, 8);
    std::vector<int32_t> a = {0,1,-1,-1,-1,-1,-1,-1}, b = {2,3,-1,-1,-1,-1,-1,-1};
    std::vector<int32_t *> maps = {a.data(), b.data()};
    std::vector<int32_t> keys = {0,1,8,9};
    const auto original_a = a, original_b = b, original_keys = keys;
    for (int i = 0; i < 64; ++i) {
        p.tick();
        const int32_t routes_a[] = {0,2}, routes_b[] = {0};
        if (!p.observe(0, routes_a, 2, a.data()) || !p.observe(1, routes_b, 1, b.data())) return false;
        if (i == 15 && !p.propose(maps, keys, 4).moves.empty()) return false;
    }
    auto q = p.propose(maps, keys, 1);
    bool ok = q.moves.size() == 1 && q.moves[0].incoming == 2 && q.moves[0].previous == 1 && q.moves[0].slot == 1;
    ok = ok && a == original_a && b == original_b && keys == original_keys;
    if (!ok) return false;
    a = q.maps[0]; b = q.maps[1]; keys = q.keys;
    maps = {a.data(), b.data()};
    p.committed(q);
    ok = a[1] == -1 && a[2] == 1 && keys[1] == 2;
    p.recent[2].heat = 0;
    for (int32_t key : {0,8,9}) p.recent[key].heat = 128;
    for (int i = 0; i < 16; ++i) {
        p.tick();
        const int32_t changed[] = {3};
        ok = p.observe(0, changed, 1, a.data()) && ok;
    }
    q = p.propose(maps, keys, 4);
    ok = ok && q.moves.empty();
    std::printf("stable routes, shadow ownership, commit and cooldown: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_cross_layer() {
    llama_moe_cache_policy p;
    p.init(2, 4);
    std::vector<int32_t> a = {0,-1,-1,-1}, b = {1,2,-1,-1};
    std::vector<int32_t *> maps = {a.data(),b.data()};
    const std::vector<int32_t> keys = {0,4,5};
    for (int i = 0; i < 64; ++i) {
        p.tick();
        const int32_t routes[] = {0,1};
        if (!p.observe(0, routes, 2, a.data())) return false;
    }
    const auto q = p.propose(maps, keys, 1);
    bool ok = q.moves.size() == 1 && q.moves[0].incoming == 1 && q.moves[0].previous == 4 && q.moves[0].slot == 1;
    ok = ok && q.maps[0][1] == 1 && q.maps[1][0] == -1 && q.maps[1][1] == 2;
    ok = ok && a[1] == -1 && b[0] == 1 && keys[1] == 4;
    std::printf("cross-layer victim invalidation and coverage floor: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_one_off() {
    llama_moe_cache_policy p;
    p.init(1, 128);
    std::vector<int32_t> map(128, -1), keys = {0,1};
    map[0] = 0; map[1] = 1;
    bool ok = true;
    for (int32_t i = 0; i < 96; ++i) {
        p.tick();
        const int32_t one_off[] = {i+2,i+2};
        ok = p.observe(0, one_off, 2, map.data()) && ok;
        ok = p.propose({map.data()}, keys, 4).moves.empty() && ok;
    }
    const auto before = p.recent;
    const int32_t invalid[] = {2,128};
    ok = !p.observe(0, invalid, 2, map.data()) && !p.observe(-1, invalid, 1, map.data()) && ok;
    for (int i = 0; i < 16; ++i) p.tick();
    ok = p.propose({map.data()}, keys, 0).moves.empty() && p.considered == p.step && ok;
    for (size_t i = 0; i < before.size(); ++i) ok = ok && before[i].heat == p.recent[i].heat && before[i].misses == p.recent[i].misses;
    map[0] = 1;
    for (int i = 0; i < 16; ++i) p.tick();
    bool rejected = false;
    try { p.propose({map.data()}, keys, 4); } catch (const std::runtime_error &) { rejected = true; }
    ok = ok && rejected;
    std::printf("one-off/duplicate routes and corrupt-map rejection: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static bool check_shifts() {
    llama_moe_cache_policy p;
    p.init(1, 8);
    std::vector<int32_t> map(8, -1), keys = {-1,-1};
    size_t promotions = 0;
    bool ok = true;
    for (int32_t target = 2; target < 7; ++target) {
        const size_t previous = promotions;
        for (int i = 0; i < 96; ++i) {
            p.tick();
            const int32_t ids[] = {0,target};
            ok = p.observe(0, ids, 2, map.data()) && ok;
            const auto q = p.propose({map.data()}, keys, 1);
            promotions += q.moves.size();
            if (!q.moves.empty()) {
                map = q.maps[0];
                keys = q.keys;
                p.committed(q);
            }
            for (size_t s = 0; s < keys.size(); ++s) ok = ok && (keys[s] < 0 || map[keys[s]] == int32_t(s));
        }
        ok = ok && map[0] >= 0 && map[target] >= 0 && promotions - previous <= 2;
    }
    ok = ok && promotions == 6;
    std::printf("empty-slot bootstrap, topic shifts and repeated slot reuse: %s (%zu promotions)\n", ok ? "PASS" : "FAIL", promotions);
    return ok;
}

int main() {
    const bool replay = check_replay();
    const bool cross_layer = check_cross_layer();
    const bool one_off = check_one_off();
    const bool shifts = check_shifts();
    return replay && cross_layer && one_off && shifts ? 0 : 1;
}
