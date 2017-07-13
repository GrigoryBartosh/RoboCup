#pragma once

#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

struct SingleImage
{
private:
    static size_t&  W()          { static size_t  _W  = 0;            return _W;     }
    static size_t&  H()          { static size_t  _H  = 0;            return _H;     }
    static size_t&  CW()         { static size_t  _CW = 0;            return _CW;    }
    static size_t&  CH()         { static size_t  _CH = 0;            return _CH;    }
    static size_t&  count()      { static size_t  _count = 0;         return _count; }
    static cv::Mat& single_img() { static cv::Mat _single_img; return _single_img;   }

public:
    SingleImage() = delete;

    static void clear()
    {
        count() = 0;
        single_img().setTo(cv::Scalar(0,0,0));
    }

    static void init(size_t W_, size_t H_, size_t CW_, size_t CH_)
    {
        W() = W_;
        H() = H_;
        CW() = CW_;
        CH() = CH_;
        single_img().create(cv::Size(W() * CW(), H() * CH()), CV_8UC3);
        clear();
    }

    static void add(const cv::Mat& img)
    {
        if (count() == CW() * CH()) return;

        img.copyTo(single_img()(cv::Rect(W() * (count() % CW()), H() * (count() / CW()), W(), H())));
        count()++;
    }

    static void show()
    {
        cv::imshow("img", single_img());
    }
};

template<class T>
struct Field
{
private:
    size_t _w, _h;
    std::vector<T> _field;

public:
    Field(const size_t w = 0, const size_t h = 0)
    :_w(w), _h(h)
    { _field.assign(_w * _h, T()); }
    Field& operator=(const Field& other)
    {
        _w = other._w;
        _h = other._h;
        _field = other._field;
        return *this;
    }
    Field(const Field& other) { *this = other; }
    Field(Field&& other)
    {
        std::swap(_w, other._w);
        std::swap(_h, other._h);
        std::swap(_field, other._field);
    }

    size_t width()  const { return _w; }
    size_t height() const { return _h; }

    T& operator()(size_t x, size_t y) 
    {
        assert(x < width() && y < height() && "bad coordinate");
        return _field[width() * y + x]; 
    }
    T operator()(size_t x, size_t y) const 
    {
        assert(x < width() && y < height() && "bad coordinate");
        return _field[width() * y + x]; 
    }

    T& operator()(cv::Point p) 
    {
        assert(p.x < static_cast<int>(width()) && p.y < static_cast<int>(height()) && "bad coordinate");
        return _field[width() * p.y + p.x]; 
    }
    T operator()(cv::Point p) const 
    {
        assert(p.x < static_cast<int>(width()) && p.y < static_cast<int>(height()) && "bad coordinate");
        return _field[width() * p.y + p.x]; 
    }

    void fill(T v)
    {
        for (size_t i = 0; i < _field.size(); i++)
            _field[i] = v;
    }

    template<class TO>
    Field<TO> convertTo()
    {
        Field<TO> f(width(), height());
        for (size_t x = 0; x < width() ; x++)
        for (size_t y = 0; y < height(); y++)
            f(x, y) = static_cast<TO>((*this)(x, y));
        return f;
    }
};

template<>
struct Field<Eigen::Vector3d>
{
private:
    typedef std::vector<Eigen::Vector3d,Eigen::aligned_allocator<Eigen::Vector3d> > vectorV3;

    size_t _w, _h;
    vectorV3 _field;

public:
    Field(const size_t w = 0, const size_t h = 0)
    :_w(w), _h(h)
    { _field.assign(_w * _h, Eigen::Vector3d::Zero()); }
    Field& operator=(const Field& other)
    {
        _w = other._w;
        _h = other._h;
        _field = other._field;
        return *this;
    }
    Field(const Field& other) { *this = other; }
    Field(Field&& other)
    {
        std::swap(_w, other._w);
        std::swap(_h, other._h);
        std::swap(_field, other._field);
    }

    size_t width()  const { return _w; }
    size_t height() const { return _h; }

    Eigen::Vector3d& operator()(size_t x, size_t y) 
    {
        assert(x < width() && y < height() && "bad coordinate");
        return _field[width() * y + x]; 
    }
    Eigen::Vector3d operator()(size_t x, size_t y) const 
    {
        assert(x < width() && y < height() && "bad coordinate");
        return _field[width() * y + x]; 
    }

    Eigen::Vector3d& operator()(cv::Point p) 
    {
        assert(p.x < static_cast<int>(width()) && p.y < static_cast<int>(height()) && "bad coordinate");
        return _field[width() * p.y + p.x]; 
    }
    Eigen::Vector3d operator()(cv::Point p) const 
    {
        assert(p.x < static_cast<int>(width()) && p.y < static_cast<int>(height()) && "bad coordinate");
        return _field[width() * p.y + p.x]; 
    }

    void fill(Eigen::Vector3d v)
    {
        for (size_t i = 0; i < _field.size(); i++)
            _field[i] = v;
    }
};

template<>
struct Field<bool>
{
private:
    size_t _w, _h;
    std::vector<char> _field;

public:
    Field(const size_t w = 0, const size_t h = 0)
    :_w(w), _h(h)
    { _field.assign(_w * _h, false); }
    Field& operator=(const Field& other)
    {
        _w = other._w;
        _h = other._h;
        _field = other._field;
        return *this;
    }
    Field(const Field& other) { *this = other; }
    Field(Field&& other)
    {
        std::swap(_w, other._w);
        std::swap(_h, other._h);
        std::swap(_field, other._field);
    }

    size_t width()  const { return _w; }
    size_t height() const { return _h; }

    char& operator()(size_t x, size_t y) 
    {
        assert(x < width() && y < height() && "bad coordinate");
        _field[width() * y + x] = (bool)_field[width() * y + x];
        return _field[width() * y + x]; 
    }
    bool operator()(size_t x, size_t y) const 
    {
        assert(x < width() && y < height() && "bad coordinate");
        return _field[width() * y + x]; 
    }

    char& operator()(cv::Point p) 
    {
        assert(p.x < static_cast<int>(width()) && p.y < static_cast<int>(height()) && "bad coordinate");
        _field[width() * p.y + p.x] = (bool)_field[width() * p.y + p.x];
        return _field[width() * p.y + p.x]; 
    }
    bool operator()(cv::Point p) const 
    {
        assert(p.x < static_cast<int>(width()) && p.y < static_cast<int>(height()) && "bad coordinate");
        return _field[width() * p.y + p.x]; 
    }

    void fill(bool v)
    {
        for (size_t i = 0; i < _field.size(); i++)
            _field[i] = v;
    }
};

inline std::tuple<cv::Scalar, cv::Scalar> boards_by_hist(const cv::Mat& img, const Field<bool>& mask, size_t percent)
{
    assert(img.type() == CV_8UC3 && percent <= 100 && "bad input");

    const size_t W = img.cols;
    const size_t H = img.rows;

    size_t sum = 0;
    size_t hist[3][256];
    memset(hist, 0, sizeof(hist));

    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (!mask(x, y)) continue;

        sum++;
        for (size_t i = 0; i < 3; i++)
            hist[i][ img.at<cv::Vec3b>(cv::Point(x, y))[i] ]++;
    }

    cv::Scalar board_down, board_up;
    for (size_t i = 0; i < 3; i++)
    {
        size_t best_dist = 256;
        for (size_t l = 0, r = 0, lsum = 0; r < 256; r++)
        {
            lsum += hist[i][r];
            while (l < r && sum * percent < lsum * 100.0) 
            {
                if (best_dist > r - l)
                {
                    best_dist = r - l;
                    board_down[i] = l;
                    board_up[i]   = r;
                }
                lsum -= hist[i][l++];
            }
        }
    }

    return std::make_tuple(board_down, board_up);
}

inline bool check_color_in_boards(cv::Scalar color, cv::Scalar board_down, cv::Scalar board_up)
{
    return board_down[0] <= color[0] && color[0] <= board_up[0] &&
           board_down[1] <= color[1] && color[1] <= board_up[1] &&
           board_down[2] <= color[2] && color[2] <= board_up[2];
}