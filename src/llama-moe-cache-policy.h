#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct llama_moe_cache_policy {
    struct usage {
        uint32_t heat = 0;
        uint32_t misses = 0;
        uint64_t admitted = 0;
    };

    struct move {
        int32_t slot;
        int32_t previous;
        int32_t incoming;
    };

    struct plan {
        std::vector<std::vector<int32_t>> maps;
        std::vector<int32_t> keys;
        std::vector<move> moves;
    };

    int32_t n_expert = 0;
    uint64_t step = 0;
    uint64_t considered = 0;
    std::vector<usage> recent;

    void init(size_t n_layer, int32_t experts) {
        if (experts < 1 || n_layer > (1u << 20)/size_t(experts)) throw std::runtime_error("adaptive expert history is too large");
        n_expert = experts;
        recent.resize(n_layer*experts);
    }

    void tick() {
        ++step;
        if (step % 32 == 0) {
            for (auto & u : recent) {
                u.heat /= 2;
                u.misses /= 2;
            }
        }
    }

    bool observe(int32_t layer, const int32_t * ids, size_t count, const int32_t * slots) {
        if (n_expert < 1 || layer < 0 || size_t(layer) >= recent.size()/n_expert || !ids || !slots) return false;
        for (size_t i = 0; i < count; ++i) if (ids[i] < 0 || ids[i] >= n_expert) return false;
        for (size_t i = 0; i < count; ++i) {
            bool duplicate = false;
            for (size_t j = 0; j < i; ++j) duplicate |= ids[i] == ids[j];
            if (duplicate) continue;
            auto & u = recent[size_t(layer)*n_expert + ids[i]];
            u.heat = std::min(u.heat + 1, 65535u);
            if (slots[ids[i]] < 0) u.misses = std::min(u.misses + 1, 65535u);
        }
        return true;
    }

    plan propose(const std::vector<int32_t *> & live, const std::vector<int32_t> & keys, size_t limit) {
        plan p;
        if (step - considered < 16) return p;
        considered = step;
        if (!limit) return p;
        if (n_expert < 1 || live.size() != recent.size()/n_expert) throw std::runtime_error("invalid adaptive layer count");
        p.maps.resize(live.size());
        p.keys = keys;
        std::vector<int32_t> coverage(live.size(), 0);
        std::vector<int32_t> candidates;
        size_t n_layer = 0;
        for (size_t il = 0; il < live.size(); ++il) {
            if (!live[il]) continue;
            ++n_layer;
            p.maps[il].assign(live[il], live[il] + n_expert);
            for (int32_t e = 0; e < n_expert; ++e) {
                const int32_t s = live[il][e], key = int32_t(il)*n_expert + e;
                if (s < -1 || (s >= 0 && (size_t(s) >= keys.size() || keys[s] != key))) throw std::runtime_error("invalid adaptive host map");
                coverage[il] += s >= 0;
                if (s < 0 && recent[key].misses >= 3 && recent[key].heat >= 4) candidates.push_back(key);
            }
        }
        for (size_t s = 0; s < keys.size(); ++s) {
            const int32_t key = keys[s];
            if (key < -1 || (key >= 0 && (size_t(key) >= recent.size() || p.maps[key/n_expert].empty() || p.maps[key/n_expert][key%n_expert] != int32_t(s)))) throw std::runtime_error("invalid adaptive slot index");
        }
        std::sort(candidates.begin(), candidates.end(), [&](int32_t a, int32_t b) {
            return recent[a].heat != recent[b].heat ? recent[a].heat > recent[b].heat : a < b;
        });
        const int32_t floor = keys.size() >= n_layer ? 1 : 0;
        for (int32_t key : candidates) {
            if (p.moves.size() >= limit) break;
            int32_t victim = -1;
            uint32_t coldest = UINT32_MAX;
            for (size_t s = 0; s < p.keys.size(); ++s) {
                const int32_t old = p.keys[s];
                if (old < 0) { victim = int32_t(s); break; }
                if (step - recent[old].admitted < 64 || (old/n_expert != key/n_expert && coverage[old/n_expert] <= floor)) continue;
                if (recent[old].heat < coldest) {
                    coldest = recent[old].heat;
                    victim = int32_t(s);
                }
            }
            if (victim < 0) continue;
            const int32_t old = p.keys[victim];
            if (old >= 0 && recent[key].heat < 4 + recent[old].heat + recent[old].heat/2) continue;
            if (old >= 0) {
                p.maps[old/n_expert][old%n_expert] = -1;
                --coverage[old/n_expert];
            }
            p.maps[key/n_expert][key%n_expert] = victim;
            ++coverage[key/n_expert];
            p.keys[victim] = key;
            p.moves.push_back({victim, old, key});
        }
        return p;
    }

    void committed(const plan & p) {
        for (const auto & m : p.moves) {
            recent[m.incoming].admitted = step;
            recent[m.incoming].misses = 0;
            if (m.previous >= 0) recent[m.previous].misses = 0;
        }
    }
};
