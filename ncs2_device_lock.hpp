#pragma once

#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ncs2 {

class DeviceBusy final : public std::runtime_error {
public:
    explicit DeviceBusy(const std::string& message) : std::runtime_error(message) {}
};

class DeviceLock {
public:
    DeviceLock() {
        const char* configured = std::getenv("NCS2_DEVICE_LOCK");
        path_ = configured && *configured
            ? configured
            : "/tmp/neuralstick-bridge-device-" + std::to_string(static_cast<unsigned long>(::getuid())) + ".lock";
        fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) {
            throw std::runtime_error("device_unavailable: cannot open device lock " + path_);
        }
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            const int error = errno;
            ::close(fd_);
            fd_ = -1;
            if (error == EWOULDBLOCK || error == EAGAIN) {
                throw DeviceBusy("device_busy: another NeuralStick Bridge process owns the device");
            }
            throw std::runtime_error("device_unavailable: cannot acquire device lock " + path_);
        }
    }

    DeviceLock(const DeviceLock&) = delete;
    DeviceLock& operator=(const DeviceLock&) = delete;

    ~DeviceLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    const std::string& path() const { return path_; }

private:
    int fd_ = -1;
    std::string path_;
};

}  // namespace ncs2
