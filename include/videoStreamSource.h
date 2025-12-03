#ifndef VIDEO_STREAM_SOURCE_H_
#define VIDEO_STREAM_SOURCE_H_

namespace surface_reconstruction {
class VideoStreamSource {
public:
    virtual void get_frame();

private:
};
}  // namespace surface_reconstruction
#endif
