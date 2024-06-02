//
// Test GroupKV
//

#include "groupkv.hpp"


static void gkv_inited() {
    GroupKV gkv = {{
        {"Group1",{
            {"testkey11", 11},
            {"testkey12", true}
        }},
        {"Group2", {
            {"key21", "val21"},
            {"key22", 22},
            {"key23", 2.3}
        }}
    }};

    std::cout << "**** gkv inited **** " << std::endl;
    std::cout << gkv.dump("", "INFO:GKV:Inited") << std::endl;

}

static void gkv_set() {

    std::cout << "**** gkv set **** " << std::endl;
    GroupKV gkv = {{}};
    std::cout << gkv.dump("", "INFO:GKV:Set:Initial") << std::endl;

    gkv.get_value("testme", {"key101b"}, false);
    gkv.get_value<std::string>("testme", {"key101s"}, "Not found");
    gkv.get_value("testme", {"key101i"}, 123456);
    gkv.get_value("testme", {"key101d"}, 123456.789);

    gkv.set_value("testme", {"key201b"}, true);
    gkv.set_value("testme", {"key201s"}, "hello world");
    gkv.set_value("testme", {"key201i"}, 987654);
    gkv.set_value("testme", {"key201d"}, 9988.7766);

    std::cout << gkv.dump("testme", "INFO:GKV:Set:After testme set") << std::endl;
    gkv.get_value("testme", {"key201b"}, false);
    gkv.get_value<std::string>("testme", {"key201s"}, "Not found");
    gkv.get_value("testme", {"key201i"}, 123456);
    gkv.get_value("testme", {"key201d"}, 123456.789);

    gkv.get_vector<int64_t>("testme", {"keyA100"}, {1, 2, 3});
    gkv.get_vector<std::string>("testme", {"keyA100"}, { "A", "അ", "अ", "ಅ" });
    gkv.set_value("testme", {"keyA300-0"}, 330);
    gkv.set_value("testme", {"keyA300-1"}, 331);
    gkv.set_value("testme", {"keyA300-2"}, 332);
    gkv.set_value("testme", {"keyA301-0"}, "India");
    gkv.set_value<std::string>("testme", {"keyA301", "1"}, "World");
    gkv.set_value("testme", {"keyA301", "2"}, "AkashaGanga");
    gkv.get_vector<int32_t>("testme", {"keyA300"}, {1, 2, 3});
    gkv.get_vector<std::string>("testme", {"keyA301"}, { "yes 1", "No 2", "very well 3" });
}

int main(int argc, char **argv) {
    log_set_target(log_filename_generator("chaton-groupkv", "log"));
    log_dump_cmdline(argc, argv);
    gkv_inited();
    gkv_set();
    return 0;
}
