#include "ObjectRecognizer.h"

using std::vector;
using std::string;
using std::set;
using std::map;
using std::unordered_map;
using std::ifstream;
using std::tuple;

using std::min;
using std::swap;
using std::make_tuple;
using std::tie;

using cv::Point;
using cv::Point2f;

using Eigen::Vector3d;
using Eigen::Vector2d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Affine2d;

ObjectRecognizer::Plane::Plane()
{
    _normal = Vector3d::Zero();
    _D = 0;
}

ObjectRecognizer::Plane::Plane(Vector3d p1, Vector3d p2, Vector3d p3)
{
    Vector3d v1 = p1 - p2;
    Vector3d v2 = p3 - p2;
    _normal = v1.cross(v2).normalized();
    _D = -_normal.dot(p1);
}

void ObjectRecognizer::Plane::set_normal(Vector3d n)
{
    _normal = n.normalized();
}

void ObjectRecognizer::Plane::set_D(double D)
{
    _D = D;
}

double ObjectRecognizer::Plane::dist(Vector3d v) const
{
    return abs(_normal.dot(v) + _D);
}

double ObjectRecognizer::Plane::cos_angle(Vector3d n) const
{
    return n.normalized().dot(_normal);
}

vector<Point> ObjectRecognizer::Pattern::read(string file_name)
{
    ifstream fin(file_name,  std::ios_base::binary);

    size_t n;
    fin.read(reinterpret_cast<char*>(&n), sizeof(size_t));

    vector<Point> points;
    for (size_t i = 0; i < n; i++)
    {
        Point p;
        fin.read(reinterpret_cast<char*>(&p.x), sizeof(int));
        fin.read(reinterpret_cast<char*>(&p.y), sizeof(int));
        points.push_back(p);
    }

    fin.close();

    return points;
}

vector<Point> ObjectRecognizer::Pattern::get_contours(const vector<Point>& points)
{
    Field<bool> mask(W, H);
    for (Point p : points)
        mask(p) = true;

    vector<Point> contours;
    for (Point p : points)
    {
        size_t neighbors = 0;
        for (size_t d = 0; d < 4; d++)
        {
            int tx = p.x + dx(d);
            int ty = p.y + dy(d);
            if (tx < 0 || tx >= static_cast<int>(W) || ty < 0 || ty >= static_cast<int>(H)) continue;
            
            neighbors += !mask(tx, ty);
        }

        if (neighbors != 0)
            contours.push_back(p);
    }

    return contours;
}

void ObjectRecognizer::Pattern::init_voronoi(const vector<Point>& points)
{
    vector<Point> contours = get_contours(points);

    cv::Rect rect(0, 0, W, H);
    cv::Subdiv2D subdiv(rect);

    for (Point p : contours) 
        subdiv.insert(p);

    vector<vector<Point2f>> facets;
    vector<Point2f> centers;
    subdiv.getVoronoiFacetList(vector<int>(), facets, centers);

    vector<Point> ifacet;
    vector<vector<Point> > ifacets(1);

    cv::Mat img = cv::Mat::zeros(cv::Size(W, H), CV_8UC3);
    for (size_t i = 0; i < facets.size(); i++)
    {
        ifacet.resize(facets[i].size());
        for(size_t j = 0; j < facets[i].size(); j++)
            ifacet[j] = facets[i][j];
 
        cv::Scalar color;
        color[0] = i % 256;
        color[1] = (i / 256) % 256;
        color[2] = (i / 256 / 256) % 256;
        cv::fillConvexPoly(img, ifacet, color, 8, 0);
    }

    _field = Field<Point>(W, H);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        size_t num = 0;
        num += img.at<cv::Vec3b>(Point(x, y))[0];
        num += img.at<cv::Vec3b>(Point(x, y))[1] * 256;
        num += img.at<cv::Vec3b>(Point(x, y))[2] * 256 * 256;
        _field(x, y) = contours[num];
    }
}

void ObjectRecognizer::Pattern::init_all(const vector<Point>& points)
{
    _center = Point(0, 0);
    for (Point p : points)
    {
        _field(p) = p;
        _center = Point(_center.x + p.x, _center.y + p.y);
    }
    _center = Point(_center.x / points.size(), _center.y / points.size());
}

Affine2d ObjectRecognizer::Pattern::make_transform(double a, double tx, double ty)
{
    Affine2d T;
    T.matrix() << cos(a), -sin(a), tx, 
                  sin(a),  cos(a), ty,
                       0,       0,  1;
    return T;
}

Affine2d ObjectRecognizer::Pattern::icp_iteration(const vector<Point>& object, const Affine2d& T)
{
    static vector<Point> actual_points;
    actual_points.clear();
    for (Point p : object)
    {
        Vector2d p2, p1;
        p2 << p.x, p.y;
        p2 = T * p2;
        if ((int)p2.x() < 0 || (int)p2.x() >= static_cast<int>(W) || (int)p2.y() < 0 || (int)p2.y() >= static_cast<int>(H)) continue;
        Point p1_cv = _field(p2.x(), p2.y());
        p1 << p1_cv.x, p1_cv.y;
        if ((p1 - p2).norm() < 1.5) continue;

        actual_points.push_back(p);
    }

    if (actual_points.size() == 0) return make_transform(0,0,0);

    size_t n = actual_points.size();
    MatrixXd A(2*n, 3);
    VectorXd b(2*n);
    for (size_t i = 0; i < n; i++)
    {
        Vector2d p2, p1;
        p2 << actual_points[i].x, actual_points[i].y;
        p2 = T * p2;
        if ((int)p2.x() < 0 || (int)p2.x() >= static_cast<int>(W) || (int)p2.y() < 0 || (int)p2.y() >= static_cast<int>(H)) continue;
        Point p1_cv = _field(p2.x(), p2.y());
        p1 << p1_cv.x, p1_cv.y;

        A(2 * i,     0) = -p2.y();
        A(2 * i,     1) = 1;
        A(2 * i,     2) = 0;
        A(2 * i + 1, 0) = p2.x();
        A(2 * i + 1, 1) = 0;
        A(2 * i + 1, 2) = 1;
        b(2 * i)        = p1.x() - p2.x();
        b(2 * i + 1)    = p1.y() - p2.y();
    }

    Vector3d x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    return make_transform(x(0), x(1), x(2));
}

bool ObjectRecognizer::Pattern::isMinorTransformation(const Affine2d& T)
{
    return abs(T.matrix()(0, 0) - 1) < EPS() && abs(T.matrix()(0, 2)) < 1.5 && abs(T.matrix()(1, 2)) < 1.5;
}

double ObjectRecognizer::Pattern::get_fitness(const vector<Point>& object, const Affine2d& T)
{
    Field<int> sum(W, H);

    for (Point p : _points)
        sum(p)++;

    size_t cnt = 0;
    for (Point p : object)
    {
        Vector2d v;
        v << p.x, p.y;
        v = T * v;
        Point cvp = Point(v.x(), v.y());
        if (cvp.x < 0 || cvp.x >= static_cast<int>(W) || cvp.y < 0 || cvp.y >= static_cast<int>(H))
        {
            cnt++;
            continue;
        }
        sum(cvp)++;
    }

    
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        if (sum(x, y) == 1) cnt++;

    return (double) cnt / _points.size();
}

void ObjectRecognizer::Pattern::draw(const vector<Point>& object, const Affine2d T)
{
    cv::Mat img(cv::Size(W, H), CV_8UC3);
    img.setTo(cv::Scalar(0,0,0));

    for (Point p : _points)
        img.at<cv::Vec3b>(p)[0] = 255;

    for (Point p : object)
    {
        Vector2d v;
        v << p.x, p.y;
        v = T * v;
        Point cvp = Point(v.x(), v.y());
        if (cvp.x < 0 || cvp.x >= static_cast<int>(W) || cvp.y < 0 || cvp.y >= static_cast<int>(H)) continue;
        
        img.at<cv::Vec3b>(cvp)[2] = 255;
    }

    cv::imshow("tmp", img);
    cv::waitKey(1);
}

ObjectRecognizer::Pattern::Pattern(string file_name)
{
    init(file_name);
}

void ObjectRecognizer::Pattern::init(string file_name)
{
    _points = read(file_name);
    init_voronoi(_points);
    init_all(_points);
}

tuple<double, double> ObjectRecognizer::Pattern::compare(const vector<Point>& object, const Point center)
{
    size_t n = object.size();
    MatrixXd A(2 * n, 3);
    VectorXd b(2 * n), x(n);

    double best_fitness = INF;
    Affine2d best_T;
    for (size_t itr = 0; itr < ICP_ITR; itr++)
    {
        Affine2d T = make_transform(0, _center.x, _center.y) * 
                     make_transform(2.0 * M_PI * rand() / ((double)RAND_MAX + 1), 0, 0) * 
                     make_transform(0, -center.x, -center.y);

        Affine2d T_new;
        size_t itr_cnt = 0;
        do
        {
            //draw(object, T);
            T_new = icp_iteration(object, T);
            T = T_new * T;
            itr_cnt++;
        } while (!isMinorTransformation(T_new) && itr_cnt < ICP_MAX_ITR);

        double fitness = get_fitness(object, T);
        if (best_fitness > fitness)
        {
            best_fitness = fitness;
            best_T = T;
        }
    }

    double angle = atan2(best_T.matrix()(1,0), best_T.matrix()(0,0));
    while (angle >  M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;

    return make_tuple(best_fitness, angle);
}

Field<Vector3d> ObjectRecognizer::make_cloud(const RealCam& cam)
{
    Field<Vector3d> cloud(W, H);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        cloud(x, y) = cam.depth(x, y).world;
    return cloud;
}

Field<Vector3d> ObjectRecognizer::calc_normals(const Field<Vector3d>& cloud)
{
    Field<Vector3d> normals(W, H);
    for (size_t x = 1; x < W-1; x++)
    for (size_t y = 1; y < H-1; y++)
    {
        if (cloud(x, y).norm() < EPS()) continue;
        if (cloud(x-1, y).norm() < EPS()) continue;
        if (cloud(x+1, y).norm() < EPS()) continue;
        if (cloud(x, y-1).norm() < EPS()) continue;
        if (cloud(x, y+1).norm() < EPS()) continue;
        
        Vector3d v_u = cloud(x+1, y) - cloud(x-1, y);
        Vector3d v_r = cloud(x, y+1) - cloud(x, y-1);

        if (v_u.norm() > NORMALS_MAX_DIST()) continue;
        if (v_r.norm() > NORMALS_MAX_DIST()) continue;

        normals(x, y) = v_r.cross(v_u).normalized();
    }
    return normals;
}

bool ObjectRecognizer::check_point_in_plane(const Plane& plane, Vector3d p, Vector3d normal)
{
    return plane.dist(p) < PLANE_BOARD_DIST() && plane.cos_angle(normal) > PLANE_BOARD_ANGLE();
}

Field<bool> ObjectRecognizer::get_mask_by_plane(const vector<Point>& good_points, const Field<Vector3d>& cloud, const Field<Vector3d>& normals,
                                                const Plane plane)
{
    Field<bool> mask(W, H);
    for (Point pix : good_points)
    {
        Vector3d p = cloud(pix);
        Vector3d n = normals(pix);
        if (!check_point_in_plane(plane, p, n)) continue;

        mask(pix) = true;
    }

    return mask;
}

void ObjectRecognizer::convert_mask2color(Field<bool>& mask, const RealCam& cam)
{
    Field<bool> mask_new(W, H);

    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (!mask(x, y)) continue;
        if (!cam.depth(x, y).connect) continue;
        mask_new(cam.depth(x, y).pixel) = true;
    }
    mask = mask_new;
}

void ObjectRecognizer::companents_filter(Field<bool>& mask, const bool val)
{
    Field<int> cmp(W, H);
    cmp.fill(-1);
    size_t num = 0;

    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (mask(x, y) != val) continue;
        if (cmp(x, y) != -1) continue;

        vector<Point> Q = {Point(x, y)};
        cmp(x, y) = num;
        for (size_t i = 0; i < Q.size(); i++)
        {
            Point cur = Q[i];

            for (size_t d = 0; d < 4; d++)
            {
                Point t = Point(cur.x + dx(d), cur.y + dy(d));

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (mask(t) != val) continue;
                if (cmp(t) != -1) continue;

                Q.push_back(t);
                cmp(t) = num;
            }
        }

        num++;
        if (Q.size() < FILTER_SIZE)
        {
            for (Point p : Q)
                mask(p) = !val;
        }
    }
}

void ObjectRecognizer::companents_fan(Field<bool>& mask, size_t d, const bool fan_type)
{
    static vector<Point> Q;
    static Field<int> dist(W, H);
    Q.clear();
    dist.fill(-1);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (mask(x, y) != fan_type) continue;
    
        Q.push_back(Point(x, y));
        dist(x, y) = 0;
    }

    for (size_t i = 0; i < Q.size(); i++)
    {
        Point cur = Q[i];

        if (dist(cur) == static_cast<int>(d)) break;

        for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
        {
            Point t = Point(cur.x + dx, cur.y + dy);

            if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
            if (dist(t) != -1) continue;

            Q.push_back(t);
            dist(t) = dist(cur) + 1;
            mask(t) = fan_type;
        }
    }
}

Field<int> ObjectRecognizer::find_all_companents(const Field<bool>& mask, const bool val)
{
    static Field<int> cmp(W, H);
    cmp.fill(-1);

    size_t num = 0;
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (mask(x, y) != val) continue;
        if (cmp(x, y) != -1) continue;

        static vector<Point> Q;
        Q = {Point(x, y)};
        cmp(x, y) = num;
        for (size_t i = 0; i < Q.size(); i++)
        {
            Point cur = Q[i];

            for (size_t d = 0; d < 4; d++)
            {
                Point t = Point(cur.x + dx(d), cur.y + dy(d));

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (mask(t) != val) continue;
                if (cmp(t) != -1) continue;

                Q.push_back(t);
                cmp(t) = num;
            }
        }
        num++;
    }

    return cmp;
}

template<class T>
bool ObjectRecognizer::check_warring_neighbors(Point p, const Field<T>& mask, const T val)
{
    static const Point dd[8] = {Point(-1, -1), Point(-1,  0), Point(-1,  1), Point( 0,  1), 
                                    Point( 1,  1), Point( 1,  0), Point( 1, -1), Point( 0, -1)};

    size_t cnt = 0;
    bool state = false;
    for (int i = 0; i < 8; i++)
    {
        Point t = Point(p.x + dd[i].x, p.y + dd[i].y);

        if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) 
        {
            state = false;
            continue;
        }

        if (mask(t) == val)
        {
            if (!state)
            {
                cnt++;
                state = true;
            }
        }
        else
        {
            state = false;
        }
    }

    if (cnt <= 1) return false;
    if (cnt >  2) return true;

    Point t1 = Point(p.x + dd[0].x, p.y + dd[0].y);
    Point t2 = Point(p.x + dd[7].x, p.y + dd[7].y);

    if (t1.x < 0 || t1.x >= static_cast<int>(W) || t1.y < 0 || t1.y >= static_cast<int>(H)) return true;
    if (t2.x < 0 || t2.x >= static_cast<int>(W) || t2.y < 0 || t2.y >= static_cast<int>(H)) return true;

    return !(mask(t1) == val && mask(t2) == val);
}

Field<int> ObjectRecognizer::components_compress(const Field<bool>& mask)
{
    static Field<int> cmp;
    cmp = find_all_companents(mask);

    static unordered_map<size_t, size_t> cnt;
    cnt.clear();
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        if (cmp(x, y) != -1)
            cnt[cmp(x, y)]++;

    static vector<Point> Q;
    static Field<int> dist(W, H);
    Q.clear();
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        if (!mask(x, y))
        {
            Q.push_back(Point(x, y));
            dist(x, y) = 0;
        }

    for (size_t i = 0; i < Q.size(); i++)
    {
        Point cur = Q[i];
        
        if (dist(cur) >= static_cast<int>(COMPRESS_DIST)) continue;

        for (size_t d = 0; d < 4; d++)
        {
            Point t = Point(cur.x + dx(d), cur.y + dy(d));

            if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
            if (cmp(t) == -1) continue;
            if (cnt[cmp(t)] < COMPRESS_MIN) continue;
            if (check_warring_neighbors(t, cmp, -1)) continue;

            Q.push_back(t);
            dist(t) = dist(cur) + 1;
            cnt[cmp(t)]--;
            cmp(t) = -1;
        }
    }

    return cmp;
}

Field<int> ObjectRecognizer::find_components_plane_by_plane(const RealCam& cam, const vector<Point>& good_points, 
                                                            const Field<Vector3d>& cloud, const Field<Vector3d>& normals, const Plane plane)
{
    static Field<bool> mask;
    mask = get_mask_by_plane(good_points, cloud, normals, plane);

    companents_filter(mask, false);
    companents_fan(mask, FAN_DEPTH_PLANE, false);
    companents_filter(mask, true);

    convert_mask2color(mask, cam);
    companents_fan(mask, FAN_COLOR_PLANE, true);

    return components_compress(mask);
}

Field<int> ObjectRecognizer::find_plane(const RealCam& cam)
{
    static Field<Vector3d> cloud, normals;
    cloud = make_cloud(cam);
    normals = calc_normals(cloud);

    Plane best_plane;
    size_t best_count = 0;

    static vector<Point> good_points;
    good_points.clear();
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (normals(x, y).norm() < EPS()) continue;
        good_points.push_back(Point(x, y));
    }

    if (good_points.size() < 300) return Field<int>(W, H);

    for (size_t itr = 0; itr < PLANE_ITR; itr++)
    {
        for (int i = 0; i < 3; i++) 
            swap(good_points[i], good_points[i + rand() % (good_points.size() - i)]);

        Plane plane(cloud(good_points[0]),
                    cloud(good_points[1]),
                    cloud(good_points[2]));

        size_t count = 0;
        for (size_t i = 0; i < good_points.size(); i += 100)
        {
            Point p = good_points[i];
            if (check_point_in_plane(plane, cloud(p), normals(p)))
                count++;
        }

        if (best_count < count)
        {
            best_count = count;
            best_plane = plane;
        }
    }

    Matrix3d A = Matrix3d::Zero();
    Vector3d B = Vector3d::Zero();
    for (Point pix : good_points)
    {
        Vector3d p = cloud(pix);
        Vector3d n = normals(pix);
        if (!check_point_in_plane(best_plane, p, n)) continue;

        A(0, 0) += p.x() * p.x();
        A(0, 1) += p.x() * p.y();
        A(0, 2) += p.x() * p.z();
        A(1, 1) += p.y() * p.y();
        A(1, 2) += p.y() * p.z();
        A(2, 2) += p.z() * p.z();
        B(0) -= p.x();
        B(1) -= p.y();
        B(2) -= p.z();
    }
    A(1, 0) = A(0, 1);
    A(2, 0) = A(0, 2);
    A(2, 1) = A(1, 2);
    Vector3d normal = A.colPivHouseholderQr().solve(B);
    best_plane.set_normal(normal);
    best_plane.set_D(1 / normal.norm());

    static Field<int> plane_cmp;
    plane_cmp = find_components_plane_by_plane(  cam, good_points, cloud, normals, best_plane);
    
    static Field<bool> plane_mask(W, H);
    plane_mask.fill(false);
    for (Point pix : good_points)
    {
        Vector3d p = cloud(pix);
        Vector3d n = normals(pix);
        if (!check_point_in_plane(best_plane, p, n)) continue;

        plane_mask(pix) = true;
    }
    static cv::Mat img_depth(cv::Size(W, H), CV_8UC3);
    static cv::Mat img_normals(cv::Size(W, H), CV_8UC3);
    cam.getImageDepth(img_depth);
    draw_mask(img_depth, plane_mask, 1, 255);
    draw_normals(img_normals, normals);
    SingleImage::add(img_depth);
    SingleImage::add(img_normals);

    return plane_cmp;
}

Field<int> ObjectRecognizer::find_kernels_by_compress(const Field<int>& plane)
{
    static Field<bool> cmp_plane(W, H);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        cmp_plane(x, y) = (plane(x, y) != -1);

    static Field<bool> cmp_object;
    cmp_object = cmp_plane;
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (cmp_plane(x, y)) continue;

        static vector<Point> Q1;
        Q1 = {Point(x, y)};
        cmp_plane(x, y) = true;
        for (size_t i = 0; i < Q1.size(); i++)
        {
            Point cur = Q1[i];

            for (size_t d = 0; d < 4; d++)
            {
                Point t = Point(cur.x + dx(d), cur.y + dy(d));

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (cmp_plane(t)) continue;

                Q1.push_back(t);
                cmp_plane(t) = true;
            }
        }

        static vector<Point> Q2;
        static set<Point, Point_comparator> usd;
        Q2.clear();
        usd.clear();
        for (Point p : Q1)
        {
            for (size_t d = 0; d < 4; d++)
            {
                Point t = Point(p.x + dx(d), p.y + dy(d));

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (plane(t) == -1) continue;
                if (usd.count(t)) continue;

                Q2.push_back(t);
                usd.insert(t);
            }
        }

        for (size_t i = 0; i < Q2.size() && Q1.size() > Q2.size(); i++) //YES! All right.
        {
            Point cur = Q2[i];

            for (size_t d = 0; d < 4; d++)
            {
                Point t = Point(cur.x + dx(d), cur.y + dy(d));

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (cmp_object(t)) continue;
                if (check_warring_neighbors(t, cmp_object, true)) continue;

                Q2.push_back(t);
                cmp_object(t) = true;
            }
        }
    }

    return find_all_companents(cmp_object, false);
}

Field<int> ObjectRecognizer::segmentation(const cv::Mat& img, const Field<int>& objects_kernels, const Field<int>& plane_kernels)
{
    static vector<vector<Point>> approx;
    approx.clear();
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        int num = objects_kernels(x, y);
        if (num == -1) continue;

        if (approx.size() <= static_cast<size_t>(num)) approx.resize(num + 1);
        approx[num].push_back(Point(x, y));
    }

    size_t n = approx.size();
    static vector<map<Point, size_t, Point_comparator>> statistic;
    statistic.clear();
    statistic.resize(n);

    static cv::Mat blur;
    cv::bilateralFilter(img, blur, BILATERAL_SIZE, 75, 75);
    //cv::imshow("blur", blur);

    for (size_t itr = 0; itr < OBJECTS_ITR; itr++)
    {
        static cv::Mat markers;
        markers = cv::Mat::zeros(cv::Size(W, H), CV_32SC1);

        for (size_t i = 0; i < n; i++)
        {
            vector<Point>& iapprox = approx[i];
            size_t cnt = iapprox.size() * OBJECTS_PART / 100;
            for (size_t j = 0; j < cnt; j++)
                swap(iapprox[j], iapprox[j + rand() % (iapprox.size() - j)]);
            for (size_t j = 0; j < cnt; j++)
                markers.at<int>(iapprox[j]) = 1 + i;
        }
    
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            if (plane_kernels(x, y) != -1)
                markers.at<int>(Point(x, y)) = 1 + n + plane_kernels(x, y);

        cv::watershed(blur, markers);

        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            int num = markers.at<int>(Point(x, y));
            if (num <= 0) continue;
            if (num <= static_cast<int>(n))
                statistic[num - 1][Point(x, y)]++;
        }

        /*cv::Mat img_segments = cv::Mat::zeros(cv::Size(W, H), CV_8UC3);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (markers.at<int>(Point(x, y)) <= 0) continue;
            if (markers.at<int>(Point(x, y)) <= static_cast<int>(n))
            {
                img_segments.at<cv::Vec3b>(Point(x, y)) = cv::Vec3b(255,0,0);
            }
        }
        imshow("segments", img_segments);
        if (cvWaitKey(1) == 27) exit(0);*/
    }

    static Field<int> objects(W, H);
    objects.fill(-1);
    for (size_t i = 0; i < n; i++)
    for (auto p : statistic[i])
        if (p.second == OBJECTS_ITR)
            objects(p.first) = i;

    return objects;
}

vector<ObjectRecognizer::Object> ObjectRecognizer::recognition(const Field<int>& objects)
{
    static vector<vector<Point>> objects_points;
    objects_points.clear();
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        int num = objects(x, y);
        if (num == -1) continue;

        if (objects_points.size() <= static_cast<size_t>(num)) objects_points.resize(num + 1);
        objects_points[num].push_back(Point(x, y));
    }

    static vector<Object> recognized;
    recognized.clear();
    for (vector<Point>& points : objects_points)
    {
        Point center = Point(0, 0);
        for (Point p : points)
            center = Point(center.x + p.x, center.y + p.y);
        center = Point(center.x / points.size(), center.y / points.size());

        double best_fitness = INF;
        Object object;
        for (size_t i = 0; i < _patterns.size(); i++)
        {
            Pattern& pattern = _patterns[i];

            double fitness, angle;
            tie(fitness, angle) = pattern.compare(points, center);
            if (best_fitness > fitness)
            {
                best_fitness = fitness;
                object.angle = angle;
                object.type = static_cast<ObjectType>(i);
            }
        }
        object.pos = center;
        recognized.push_back(object);
    }

    return recognized;
}

void ObjectRecognizer::draw_normals(cv::Mat& img, const Field<Vector3d>& normals)
{
    img.setTo(cv::Scalar(0,0,0));
    for (size_t x = 1; x < W-1; x++)
    for (size_t y = 1; y < H-1; y++)
    {
        if (normals(x, y).norm() < EPS()) continue;
        img.at<cv::Vec3b>(Point(x, y)) = cv::Vec3b(255 * (normals(x, y)(0) + 1) / 2,
                                                       255 * (normals(x, y)(1) + 1) / 2, 
                                                       255 * (normals(x, y)(2) + 1) / 2);
    }
}

void ObjectRecognizer::draw_mask(cv::Mat& img, const Field<bool>& mask, size_t chennel, const unsigned char val)
{
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
    {
        if (!mask(x, y)) continue;
        cv::Vec3b color = img.at<cv::Vec3b>(Point(x, y));
        color[chennel] = val;
        img.at<cv::Vec3b>(Point(x, y)) = color;
    }
}

ObjectRecognizer::ObjectRecognizer()
{
    ifstream fin(CONFIG(), std::ios_base::in);
    string name;
    while (fin >> name)
    {
        Pattern p(name);
        _patterns.push_back(p);
    }
}

void ObjectRecognizer::get_object(const RealCam& cam)
{
    static cv::Mat img(cv::Size(W, H), CV_8UC3);
    static cv::Mat img_objects(cv::Size(W, H), CV_8UC3);

    cam.getImageColor(img);

    Field<int> plane_cmp, objects_kernels, objects;
    plane_cmp = find_plane(cam);
    objects_kernels = find_kernels_by_compress(plane_cmp);
    objects = segmentation(img, objects_kernels, plane_cmp);
    vector<Object> types = recognition(objects);

    static Field<bool> plane_cmp_mask(W, H);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        plane_cmp_mask(x, y) = (plane_cmp(x, y) != -1);

    static Field<bool> objects_kernels_mask(W, H);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        objects_kernels_mask(x, y) = (objects_kernels(x, y) != -1);

    static Field<bool> objects_mask(W, H);
    for (size_t x = 0; x < W; x++)
    for (size_t y = 0; y < H; y++)
        objects_mask(x, y) = (objects(x, y) != -1);

    img_objects.setTo(cv::Scalar(0,0,0));
    for (Object object : types)
    {
        string s = std::to_string(static_cast<int>(object.type)) + " ... " + std::to_string(object.angle / M_PI * 180);
        putText(img_objects, s, object.pos, cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255));
    }

    draw_mask(img, plane_cmp_mask, 1, 255);
    draw_mask(img, objects_kernels_mask, 0, 255);
    draw_mask(img_objects, objects_mask, 2, 255);

    SingleImage::add(img);
    SingleImage::add(img_objects);
}