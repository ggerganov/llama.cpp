
#include <iostream>
#include <sstream>
#include "mcp_sse_transport.h"
#include <log.h>
#include <chrono>

toolcall::mcp_sse_transport::~mcp_sse_transport() {
    if (endpoint_headers_) {
        curl_slist_free_all(endpoint_headers_);
    }
    if (endpoint_) {
        curl_easy_cleanup(endpoint_);
    }
}

toolcall::mcp_sse_transport::mcp_sse_transport(std::string server_uri)
    : server_uri_(std::move(server_uri)),
      running_(false),
      sse_thread_(),
      endpoint_(nullptr),
      endpoint_headers_(nullptr),
      endpoint_errbuf_(CURL_ERROR_SIZE),
      event_{"", "", ""},
      sse_buffer_(""),
      sse_cursor_(0),
      sse_last_id_(""),
      initializing_mutex_(),
      initializing_()
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

void toolcall::mcp_sse_transport::start() {
    if (running_) return;
    running_ = true;

    std::unique_lock<std::mutex> lock(initializing_mutex_);
    sse_thread_ = std::thread(&toolcall::mcp_sse_transport::sse_run, this);
    initializing_.wait(lock);

    if (endpoint_ == nullptr) {
        running_ = false;
        LOG_ERR("SSE: Connection to \"%s\" failed", server_uri_.c_str());
        throw std::runtime_error("Connection to \"" + server_uri_ + "\" failed");
    }
}

void toolcall::mcp_sse_transport::stop() {
    running_ = false;
}

bool toolcall::mcp_sse_transport::send(const std::string & request_json) {
    if (! running_ || endpoint_ == nullptr) {
        return false;
    }

    curl_easy_setopt(endpoint_, CURLOPT_POSTFIELDS, request_json.c_str());

    CURLcode code = curl_easy_perform(endpoint_);
    if (code != CURLE_OK) {
        size_t len = strlen(&endpoint_errbuf_[0]);
        LOG_ERR("%s", (len > 0 ? &endpoint_errbuf_[0] : curl_easy_strerror(code)));
        return false;
    }
    return true;
}

static size_t sse_callback(char * data, size_t size, size_t nmemb, void * clientp) {
    auto transport = static_cast<toolcall::mcp_sse_transport*>(clientp);
    size_t len = size * nmemb;
    return transport->sse_read(data, len);
}

void toolcall::mcp_sse_transport::parse_field_value(std::string field, std::string value) {
    if (field == "event") {
        // Set the event type buffer to field value.
        event_.type = std::move(value);

    } else if (field == "data") {
        // Append the field value to the data buffer,
        // then append a single U+000A LINE FEED (LF)
        // character to the data buffer.
        value += '\n';
        event_.data.insert(event_.data.end(), value.begin(), value.end());

    } else if (field == "id") {
        // If the field value does not contain U+0000 NULL,
        // then set the last event ID buffer to the field value.
        // Otherwise, ignore the field.
        if (! value.empty()) {
            event_.id = std::move(value);
        }

    } else if (field == "retry") {
        // If the field value consists of only ASCII digits,
        // then interpret the field value as an integer in base
        // ten, and set the event stream's reconnection time to
        // that integer. Otherwise, ignore the field.

        LOG_INF("SSE: Retry field is not currently implemented");

    } else {
        LOG_WRN("SSE: Unsupported field \"%s\" received", field.c_str());
    }
}

void toolcall::mcp_sse_transport::on_endpoint_event() {
    endpoint_ = curl_easy_init();
    if (! endpoint_) {
        LOG_ERR("SSE: Failed to create endpoint handle");
        running_ = false;
        return;
    }

    curl_easy_setopt(endpoint_, CURLOPT_URL, event_.data.c_str());

    endpoint_headers_ =
        curl_slist_append(endpoint_headers_, "Content-Type: application/json");
    curl_slist_append(endpoint_headers_, "Connection: keep-alive");
    curl_easy_setopt(endpoint_, CURLOPT_HTTPHEADER, endpoint_headers_);
    curl_easy_setopt(endpoint_, CURLOPT_ERRORBUFFER, &endpoint_errbuf_[0]);

    // Later calls to send will reuse the endpoint_ handle
}

void toolcall::mcp_sse_transport::on_message_event() {
    mcp::message_variant message;
    if (mcp::create_message(event_.data, message)) {
        notify_if<mcp::initialize_response>(message);
        notify_if<mcp::tools_list_response>(message);
    }
}

size_t toolcall::mcp_sse_transport::sse_read(const char * data, size_t len) {
    sse_buffer_.insert(sse_buffer_.end(), data, data + len);

    for (; sse_cursor_ < sse_buffer_.length(); ++sse_cursor_) {
        if (sse_buffer_[sse_cursor_] == '\r' || sse_buffer_[sse_cursor_] == '\n') {
            auto last = sse_buffer_.begin() + sse_cursor_;

            std::string line(sse_buffer_.begin(), last);
            if (line.empty()) { // Dispatch event
                if (event_.type == "endpoint") {
                    on_endpoint_event();

                } else if(event_.type == "message") {
                    on_message_event();

                } else {
                    LOG_WRN("SSE: Unsupported event \"%s\" received", event_.type.c_str());
                }

                sse_last_id_ = event_.id;
                event_ = {"", "", ""};

            } else if(line[0] != ':') { // : denotes a comment
                // Set field/value
                auto sep_index = line.find(':');
                if (sep_index != std::string::npos) {
                    auto sep_i = line.begin() + sep_index;

                    std::string field (line.begin(), sep_i);
                    std::string value (sep_i + 1, line.end());

                    parse_field_value(std::move(field), std::move(value));
                }
            }

            if (last++ != sse_buffer_.end()) { // Consume line-end
                if (*last == '\n') {
                    last ++; // In the CRLF case consume one more
                }
                sse_buffer_ = std::string(last, sse_buffer_.end());

            } else {
                sse_buffer_.clear();
            }
            sse_cursor_ = 0; // Prepare to scan for next line-end
        }
    }
    return len;
}

void toolcall::mcp_sse_transport::sse_run() {
    std::unique_lock<std::mutex> lock(initializing_mutex_);
    char errbuf[CURL_ERROR_SIZE];
    size_t errlen;
    CURLMcode mcode;
    int num_handles;
    struct CURLMsg * m;
    int msgs_in_queue = 0;
    CURLM * async_handle = nullptr;
    struct curl_slist * headers = nullptr;
    CURL * sse = nullptr;

    sse = curl_easy_init();
    if (! sse) {
        LOG_ERR("SSE: Failed to initialize handle");
        goto cleanup;
    }

    headers = curl_slist_append(headers, "Connection: keep-alive");

    curl_easy_setopt(sse, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(sse, CURLOPT_ERRORBUFFER, errbuf);
    curl_easy_setopt(sse, CURLOPT_URL, server_uri_.c_str());
    curl_easy_setopt(sse, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(sse, CURLOPT_WRITEFUNCTION, sse_callback);
    curl_easy_setopt(sse, CURLOPT_WRITEDATA, this);

    async_handle = curl_multi_init();
    if (! async_handle) {
        LOG_ERR("SSE: Failed to initialize async handle");
        goto cleanup;
    }
    curl_multi_add_handle(async_handle, sse);

    do {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        mcode = curl_multi_perform(async_handle, &num_handles);
        if (mcode != CURLM_OK) {
            LOG_ERR("SSE: %s", curl_multi_strerror(mcode));
            break;
        }
        while ((m = curl_multi_info_read(async_handle, &msgs_in_queue)) != nullptr) {
            if (m->msg == CURLMSG_DONE) {
                if (m->data.result != CURLE_OK) {
                    errlen = strlen(errbuf);
                    if (errlen) {
                        LOG_ERR("SSE: %s", errbuf);

                    } else {
                        LOG_ERR("SSE: %s", curl_easy_strerror(m->data.result));
                    }
                    running_ = false;
                    break;
                }
            }
        }
        if (endpoint_ && lock.owns_lock()) { // TODO: timeout if endpoint not received
            lock.unlock();
            initializing_.notify_one();
        }

    } while (running_);

  cleanup:
    if (headers) {
        curl_slist_free_all(headers);
    }
    if (async_handle) {
        curl_multi_remove_handle(async_handle, sse);
        curl_multi_cleanup(async_handle);
    }
    if (sse) {
        curl_easy_cleanup(sse);
    }
}
