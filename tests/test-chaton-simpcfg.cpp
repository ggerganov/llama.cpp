//
// Test SimpCfg
//

#include "simpcfg.hpp"

#include <iostream>
#include <format>


static void check_string() {
    std::vector<std::string> vStandard = { "123", "1अ3" };
    std::cout << "**** string **** " << vStandard.size() << std::endl;
    for(auto sCur: vStandard) {
        std::cout << std::format("string: [{}] len[{}] size[{}]", sCur, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            std::cout << std::format("string:{}:pos:{}:char:{}[0x{:x}]\n", sCur, i, c, (uint8_t)c);
            i += 1;
        }
    }
}

static void check_u8string() {
    std::vector<std::u8string> vU8s = { u8"123", u8"1अ3" };
    std::cout << "**** u8string **** " << vU8s.size() << std::endl;
    for(auto sCur: vU8s) {
        std::string sCurx (sCur.begin(), sCur.end());
        std::cout << std::format("u8string: [{}] len[{}] size[{}]", sCurx, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            //std::cout << c << std::endl;
            std::cout << std::format("u8string:{}:pos:{}:char:{}[0x{:x}]\n", sCurx, i, (unsigned char)c, (unsigned char)c);
            i += 1;
        }
    }
}

static void check_wstring_wcout() {
    std::wcout.imbue(std::locale("en_US.UTF-8"));
    std::vector<std::wstring> vWide = { L"123", L"1अ3" };
    std::cout << "**** wstring wcout **** " << vWide.size() << std::endl;
    for(auto sCur: vWide) {
        std::wcout << sCur << std::endl;
        std::wcout << std::format(L"wstring: [{}] len[{}] size[{}]", sCur, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            std::wcout << std::format(L"wstring:{}:pos:{}:char:{}[0x{:x}]\n", sCur, i, c, c);
            i += 1;
        }
    }
}

static void check_wstring_cout() {
    std::vector<std::wstring> vWide = { L"123", L"1अ3" };
    std::cout << "**** wstring cout **** " << vWide.size() << std::endl;
    for(auto sCur: vWide) {
        std::string sCury;
        wcs_to_mbs(sCury, sCur);
        std::cout << std::format("wstring: [{}] len[{}] size[{}]", sCury, sCur.length(), sCur.size()) << std::endl;
        int i = 0;
        for(auto c: sCur) {
            std::wstringstream wsc;
            wsc << c;
            std::string ssc;
            wcs_to_mbs(ssc, wsc.str());
            std::cout << std::format("wstring:{}:pos:{}:char:{}[0x{:x}]\n", sCury, i, ssc, (uint32_t)c);
            i += 1;
        }
    }
}

static void check_nonenglish() {
    std::cout << "**** non english **** " << std::endl;
    std::vector<std::string> vTest1 = { "\n\tAഅअಅ\n\t", "\n\tAഅअಅ " };
    for (auto sTest: vTest1) {
        std::string sGotDumb = str_trim_dumb(sTest, {" \n\t"});
        std::string sGotOSmart = str_trim_oversmart(sTest, {" \n\t"});
        std::string sLower = str_tolower(sTest);
        std::cout << std::format("{}: Test1 [{}]\n\tTrimDumb[{}]\n\tTrimOverSmart[{}]\n\tLowerDumb[{}]", __func__, sTest, sGotDumb, sGotOSmart, sLower) << std::endl;
    }
    // The string "\n\tthis र remove 0s and अs at end 000रअ0\xa4अ ",
    // * will mess up str_trim_dumb,
    // * but will rightly trigger a exception with oversmart.
    std::vector<std::string> vTest2 = { "\n\t this र remove 0s at end 000 ", "\n\tthis र remove 0s, अs, ഇs at end 000रअ0अ ", "\n\tthis र remove 0s, अs, ഇs at end 000रअ0इअ "};
    std::string trimChars = {" \n\tഇ0अ"};
    for (auto sTest: vTest2) {
        std::string sGotDumb = str_trim_dumb(sTest, trimChars);
        std::string sGotOSmart = str_trim_oversmart(sTest, trimChars);
        std::cout << std::format("{}: Test2 [{}]\n\tDumb[{}]\n\tOverSmart[{}]", __func__, sTest, sGotDumb, sGotOSmart) << std::endl;
    }
}

static void check_strings() {
    std::string sSavedLocale;
    SimpCfg::locale_prepare(sSavedLocale);
    check_string();
    check_u8string();
    check_wstring_wcout();
    check_wstring_cout();
    check_nonenglish();
    SimpCfg::locale_restore(sSavedLocale);
}

static void sc_inited() {
    SimpCfg sc = {{
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

    std::cout << "**** sc inited **** " << std::endl;
    std::cout << sc.dump("", "INFO:SC:Inited") << std::endl;

}

static void sc_set(const std::string &fname) {

    std::cout << "**** sc set **** " << std::endl;
    SimpCfg sc = {{}};
    sc.load(fname);
    std::cout << sc.dump("", "INFO:SC:Set:AfterLoad") << std::endl;

    sc.get_bool("testme", {"key101b"}, false);
    sc.get_string("testme", {"key101s"}, "Not found");
    sc.get_int64("testme", {"key101i"}, 123456);
    sc.get_double("testme", {"key101d"}, 123456.789);

    sc.set_bool("testme", {"key201b"}, true);
    sc.set_string("testme", {"key201s"}, "hello world");
    sc.set_int64("testme", {"key201i"}, 987654);
    sc.set_double("testme", {"key201d"}, 9988.7766);

    std::cout << sc.dump("testme", "INFO:SC:Set:AfterSet") << std::endl;
    sc.get_bool("testme", {"key201b"}, false);
    sc.get_string("testme", {"key201s"}, "Not found");
    sc.get_int64("testme", {"key201i"}, 123456);
    sc.get_double("testme", {"key201d"}, 123456.789);

    sc.get_string("mistral", {"system-prefix"}, "Not found");
    sc.get_string("\"mistral\"", {"\"system-prefix\""}, "Not found");

    sc.get_vector<int64_t>("testme", {"keyA100"}, {1, 2, 3});
    sc.get_vector<std::string>("testme", {"keyA100"}, { "A", "അ", "अ", "ಅ" });
    sc.set_int64("testme", {"keyA300-0"}, 330);
    sc.set_int64("testme", {"keyA300-1"}, 331);
    sc.set_int64("testme", {"keyA300-2"}, 332);
    sc.set_string("testme", {"keyA301-0"}, "India");
    sc.set_value<std::string>("testme", {"keyA301", "1"}, "World");
    sc.set_string("testme", {"keyA301", "2"}, "AkashaGanga");
    sc.get_vector<int64_t>("testme", {"keyA300"}, {1, 2, 3});
    sc.get_vector<std::string>("testme", {"keyA301"}, { "yes 1", "No 2", "very well 3" });
}

int main(int argc, char **argv) {
    if (argc != 2) {
        LERRR_LN("USAGE:%s simp.cfg", argv[0]);
        exit(1);
    }

    log_set_target(log_filename_generator("chaton-simpcfg", "log"));
    log_dump_cmdline(argc, argv);

    check_strings();
    sc_inited();
    std::string fname {argv[1]};
    sc_set(fname);

    return 0;
}
