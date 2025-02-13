#include <string>
#include <optional>
#include <vector>
#include "../json.hpp"

namespace mcp
{
    extern const std::string JsonRpcVersion;
    extern const std::string McpVersion;
    extern const std::string ClientVersion;
    extern const std::string ClientName;

    class message {
    public:
        message(std::optional<nlohmann::json> id = std::nullopt);

        virtual ~message() = default;
        virtual nlohmann::json toJson() const = 0;

        void id(std::optional<nlohmann::json> id);
        const std::optional<nlohmann::json> & id() const;

    private:
        std::optional<nlohmann::json> id_;
    };


    class request : public message {
    public:
        request(std::optional<nlohmann::json> id,
                std::string method,
                std::optional<nlohmann::json> params = std::nullopt);

        virtual ~request() = default;
        nlohmann::json toJson() const override;

        void method(std::string method);
        const std::string & method() const;

        void params(std::optional<nlohmann::json> params);
        const std::optional<nlohmann::json> & params() const;

    private:
        std::string method_;
        std::optional<nlohmann::json> params_;
    };


    class response : public message {
    public:
        struct error {
            int code;
            std::string message;
            std::optional<nlohmann::json> data;
            nlohmann::json toJson() const;
        };

        response(std::optional<nlohmann::json> id,
                 std::optional<nlohmann::json> result = std::nullopt,
                 std::optional<error> error = std::nullopt);

        virtual ~response() = default;
        virtual nlohmann::json toJson() const override;

        void result(std::optional<nlohmann::json> result);
        const std::optional<nlohmann::json> & result() const;

        void setError(std::optional<error> error);
        const std::optional<error> & getError() const;

    private:
        std::optional<nlohmann::json> result_;
        std::optional<error> error_;
    };


    class notification : public message {
    public:
        notification(std::string method,
                     std::optional<nlohmann::json> params = std::nullopt);

        virtual ~notification() = default;
        virtual nlohmann::json toJson() const override;

        void method(std::string method);
        const std::string & method() const;

        void params(std::optional<nlohmann::json> params);
        const std::optional<nlohmann::json> & params() const;

    private:
        std::string method_;
        std::optional<nlohmann::json> params_;
    };


    struct capability {
        std::string name;
        bool subscribe   = false;
        bool listChanged = false;
    };

    using capabilities = std::vector<capability>;

    class initialize_request : public request {
    public:
        initialize_request(nlohmann::json id, mcp::capabilities caps);

        const std::string & name()    const { return ClientName; }
        const std::string & version() const { return ClientVersion; }
        const std::string & protoVersion() const { return McpVersion; }

        void capabilities(mcp::capabilities capabilities);
        const mcp::capabilities & capabilities() const;

    private:
        void refreshParams();

        mcp::capabilities caps_;
    };


    class initialize_response : public response {
    public:
        initialize_response(nlohmann::json id,
                            std::string name,
                            std::string version,
                            std::string protoVersion,
                            mcp::capabilities caps);

        void name(std::string name);
        const std::string & name() const;

        void version(std::string version);
        const std::string & version() const;

        void protoVersion(std::string protoVersion);
        const std::string & protoVersion() const;

        void capabilities(mcp::capabilities capabilities);
        const mcp::capabilities & capabilities() const;

        static initialize_response fromJson(const nlohmann::json& j);

    private:
        void refreshResult();

        std::string name_;
        std::string version_;
        std::string protoVersion_;
        mcp::capabilities caps_;
    };


    class initialized_notification : public notification {
    public:
        initialized_notification();
    };
}

