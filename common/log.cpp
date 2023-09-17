#include "log.h"

LogStateWrapper::~LogStateWrapper()
{
    log_flush_all_targets_internal();
    for(auto t : _targets){ delete t; }
}

LogStateWrapper & LogStateWrapper::instance()
{
    static LogStateWrapper inst;
    return inst;
}

LogStateWrapper::LogTargetWrapper::LogTargetWrapper(FILE * handle)
:   _type(Type::Stream),
    _opened(true),
    _handle(handle)
{}

LogStateWrapper::LogTargetWrapper::LogTargetWrapper(const std::string && filename)
: LogTargetWrapper(filename)
{}

LogStateWrapper::LogTargetWrapper::LogTargetWrapper(const std::string & filename)
:   _filename(filename)
{
    if(_filename.empty())
    {
        _type = Type::Stream;
        _handle = stderr;
        _opened = true;
    }
    else
    {
        _type = Type::File;
    }
}

LogStateWrapper::LogTargetWrapper::~LogTargetWrapper()
{
    if(_type == Type::File && _handle != nullptr) { std::fclose(_handle); }
}

LogStateWrapper::LogTargetWrapper::operator FILE * ()
{
    if(!_opened)
    {
        while(_lock.test_and_set(std::memory_order_acquire)){ std::this_thread::yield(); }
        if(!_opened && _type == Type::File) // check for opened again after acquiring a lock
        {
            auto result = std::fopen(_filename.c_str(), "w");
            if(result)
            {
                _handle = result;
            }
            else
            {
                std::fprintf(
                    stderr,
                    "Failed to open logfile '%s' with error '%s'\n",
                    _filename.c_str(), std::strerror(errno)
                );
                std::fflush(stderr);
                _handle = stderr;
            }
            _opened = true;
        }
        _lock.clear(std::memory_order_release);
    }
    return _handle;
}

void LogStateWrapper::LogTargetWrapper::flush()
{
    while(_lock.test_and_set(std::memory_order_acquire)){ std::this_thread::yield(); }
    if(_opened && _type != Type::Invalid && _handle)
    {
        std::fflush(_handle);
    }
    _lock.clear(std::memory_order_release);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_internal(const std::string && filename)
{
    return log_set_target_internal(filename);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_internal(const std::string & filename)
{
    return log_add_select_target_internal(new LogTargetWrapper(filename), true);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_internal(FILE * handle)
{
    return log_add_select_target_internal(new LogTargetWrapper(handle), true);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_internal(LogTargetWrapper * target)
{
    return log_add_select_target_internal(target);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_add_select_target_internal(LogTargetWrapper * t, bool insert)
{
    log_flush_all_targets_internal();
    std::lock_guard<std::mutex> lock(_mutex);

    if(_global_override_target == LogTriState::True)
    {
        if(_enabled || _global_override_enabled == LogTriState::True) return _current_target;
        return _stored_target;
    }

    if(_enabled || _global_override_enabled == LogTriState::True) _current_target.store(t);
    else                                                          _stored_target.store(t);

    if(insert) _targets.insert(t);

    return t;
}

void LogStateWrapper::log_flush_all_targets_internal()
{
    std::lock_guard<std::mutex> lock(_mutex);
    for(auto t : _targets){ t->flush(); }
}

FILE * LogStateWrapper::log_handler_internal()
{
    return *_current_target;
}

FILE * LogStateWrapper::log_tee_handler_internal()
{
    return _stderr_target;
}

void LogStateWrapper::log_disable_internal(bool threadsafe)
{
    if(threadsafe)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        log_disable_internal_unsafe();
    }
    else
    {
        log_disable_internal_unsafe();
    }
}

void LogStateWrapper::log_disable_internal_unsafe()
{
    if(_enabled && _global_override_enabled != LogTriState::True)
    {
        _stored_target.store      (_current_target);
        _current_target.store     (&_null_target);
        _enabled =                false;
    }
}

void LogStateWrapper::log_enable_internal(bool threadsafe)
{
    if(threadsafe)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        log_enable_internal_unsafe();
    }
    else
    {
        log_enable_internal_unsafe();
    }
}

void LogStateWrapper::log_enable_internal_unsafe()
{
    if(!_enabled && _global_override_enabled != LogTriState::False)
    {
        _current_target.store     (_stored_target);
        _enabled =                true;
    }
}

bool LogStateWrapper::log_param_single_parse_internal(const std::string & param)
{
#ifdef LOG_WITH_TEST
    if (param == "--log-test")
    {
        log_test();
        return true;
    }
#endif
    if (param == "--log-disable")
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            log_disable_internal_unsafe();
            _global_override_enabled = LogTriState::False;
        }
        return true;
    }

    if (param == "--log-enable")
    {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            log_enable_internal_unsafe();
            _global_override_enabled = LogTriState::True;
        }
        return true;
    }

    return false;
}

bool LogStateWrapper::log_param_pair_parse_internal(bool parse, const std::string & param, const std::string & next)
{
    if (param == "--log-file")
    {
        if (parse)
        {
            log_flush_all_targets_internal();
            std::lock_guard<std::mutex> lock(_mutex);
            auto t = new LogTargetWrapper(log_filename_generator(next.empty() ? "unnamed" : next, "log"));
            if(_enabled)
            {
                _current_target.store(t);
            }
            else
            {
                _stored_target.store(t);
            }
            _targets.insert(t);
            _global_override_target = LogTriState::True;
            return t;
        }

        return true;
    }

    return false;
}

std::string LogStateWrapper::log_filename_generator_internal(const std::string & basename, const std::string & extension)
{
    std::stringstream buf;

    buf << basename;
    buf << ".";
    buf << log_get_pid_impl();
    buf << ".";
    buf << extension;

    return buf.str();
}

std::string LogStateWrapper::log_get_pid_internal()
{
    static std::string pid;
    if (pid.empty())
    {
        // std::this_thread::get_id() is the most portable way of obtaining a "process id"
        //  it's not the same as "pid" but is unique enough to solve multiple instances
        //  trying to write to the same log.
        std::stringstream ss;
        ss << std::this_thread::get_id();
        pid = ss.str();
    }

    return pid;
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_impl(const std::string && filename)
{
    return log_set_target_impl(filename);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_impl(const std::string & filename)
{
    return instance().log_set_target_internal(filename);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_impl(FILE * handle)
{
    return instance().log_set_target_internal(handle);
}

LogStateWrapper::LogTargetWrapper * LogStateWrapper::log_set_target_impl(LogTargetWrapper * target)
{
    return instance().log_set_target_internal(target);
}

FILE * LogStateWrapper::log_handler_impl()
{
    return instance().log_handler_internal();
}

FILE * LogStateWrapper::log_tee_handler_impl()
{
    return instance().log_tee_handler_internal();
}

void LogStateWrapper::log_disable_impl()
{
    instance().log_disable_internal();
}

void LogStateWrapper::log_enable_impl()
{
    instance().log_enable_internal();
}

bool LogStateWrapper::log_param_single_parse_impl(const std::string & param)
{
    return instance().log_param_single_parse_internal(param);
}

bool LogStateWrapper::log_param_pair_parse_impl(bool parse, const std::string & param, const std::string & next)
{
    return instance().log_param_pair_parse_internal(parse, param, next);
}

std::string LogStateWrapper::log_filename_generator_impl(const std::string & basename, const std::string & extension)
{
    return instance().log_filename_generator_internal(basename, extension);
}

std::string LogStateWrapper::log_get_pid_impl()
{
    return instance().log_get_pid_internal();
}
