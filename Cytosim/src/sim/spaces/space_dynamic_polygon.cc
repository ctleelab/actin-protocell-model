// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
// Note: membrane deformation parameter (/mu_membrane) is not robust to changes in tension and bending rigidity
// The equilibrium radius for a given tension and bending depends on the membrane deformation parameter
// Early testing suggests some benchmarking to find the right membrane deformation parameter for desired polygon radius at each set of tension and bending parameters


#include "dim.h"
#include "space_dynamic_polygon.h"
#include "space_dynamic_prop.h"
#include "exceptions.h"
#include "mecapoint.h"
#include "glossary.h"
#include "polygon.h"
#include "random.h"
#include "meca.h"
#include <fstream>
#include <numeric>
#include <cmath>
#include <vector>


SpaceDynPolygon::SpaceDynPolygon(SpaceDynamicProp const* p)
: Space(p)
{
    inf_.reset();
    sup_.reset();
    surface_ = 0;
    height_ = 0;
    reset_forces();
   
    if ( DIM == 1 )
        throw InvalidParameter("polygon is not usable in 1D");
}


SpaceDynPolygon::~SpaceDynPolygon()
{
}

//------------------------------------------------------------------------------
#pragma mark - I/O

/**
 recalculate bounding box, volume
 and points offsets that are used to project
 */
void SpaceDynPolygon::resize(Glossary& opt)
{
    unsigned ord = 40;
    std::string file;
    
    if ( opt.set(file, "file") )
        poly_.read(file);
    else if ( !prop()->dimensions.empty() )
        poly_.read(prop()->dimensions);
    else if ( opt.has_key("points") )
    {
        // specify vertices directly:
        unsigned nbp = (unsigned)opt.num_values("points");
        poly_.allocate(nbp);
        for ( unsigned p = 0; p < nbp; ++p )
        {
            Vector2 vec(0,0);
            if ( ! opt.set(vec, "points", p) )
                throw InvalidParameter("polygon:points must be a list of comma-separated points: X Y, X Y, X Y, etc.");
            poly_.setPoint(p, vec.XX, vec.YY);
        }

        resize_force(nbp); // resize forces to fit number of polygon vertices
    }
    else if ( opt.set(ord, "order") )
    {
        real rad, ang = 0;
        opt.set(rad, "radius");
        opt.set(ang, "angle");
        poly_.set(ord, rad, ang);

        resize_force(ord);

        //std::clog<<"polygon:order="<<ord<<", radius="<<rad<<", angle="<<ang<<std::endl;

    }
    else
        return;
    
    real x;
    if ( opt.set(x, "scale") )
        poly_.scale(x, x);

    if ( opt.set(x, "inflate") )
        poly_.inflate(x);
    
#if ( DIM == 3 )
    x = height_;
    if ( opt.set(x, "height") )
        x *= 0.5;
    if ( x < 0 )
        throw InvalidParameter("polygon:height must be >= 0");
    height_ = x;
#endif

    update();

}

void SpaceDynPolygon::resize_force(unsigned num_vertices) // Initialize with zero forces
{
    vertex_forces.resize(num_vertices);
    reset_forces();
    // for (std::size_t i = 0; i < num_vertices; ++i)
    //     {
    //         vertex_forces.push_back(Vector2(0,0));
    //     } 
}

//-----------------------------------Energy and Forces section of space dynamic polygon-------------------------------------------//
// Calculate the energy and forces from osmotic pressure acting on each vertex the polygon

real SpaceDynPolygon::_osmotic_energy() const
{   
    double Kv = 0.0; // prop()->Kv; osmotic strength in units of mN.um;
    double V_bar = 0.0; // prop()->V_bar; preferred volume in units of um^3;

    real volume = poly_.surface();
    double osmotic_energy = 0.0;

    osmotic_energy = Kv * ((volume/V_bar) - 1 - log(volume/V_bar));
    return osmotic_energy;
} 

std::vector<Vector2> SpaceDynPolygon::calculateOsmoticForces() const {

    real Kv = prop()->Kv; //osmotic strength in units of nN.um;

    real V_bar = prop()->V_bar;

    if ( Kv == 0 )
        return std::vector<Vector2>(poly_.nbPoints(), Vector2(0,0));

    real volume = abs_real(poly_.surface());
    //std::clog<<"Surface Area: "<<volume<<std::endl;

    std::vector<Vector2> osmoticForce(poly_.nbPoints());
    osmoticForce.assign(poly_.nbPoints(), Vector2(0,0));

    size_t n = poly_.nbPoints(); // vertex_positions.size();
    
    Vector2 d;
    std::vector<real> edgeLengths(n);
    std::vector<Vector2> edge_normal(n);

    for (size_t i = 0; i < n; ++i) {

        // std::clog<<"Vertex ["<<i<<"]: "<<poly_.pts_[i].xx<<", "<<poly_.pts_[i].yy<<std::endl;

        d.XX = poly_.pts_[(i + 1) % n].xx - poly_.pts_[i].xx;
        d.YY = poly_.pts_[(i + 1) % n].yy - poly_.pts_[i].yy;
        edgeLengths[i] = d.norm();

        //Calculate the edge normals. Assumes a clockwise orientation convention of the edge normal vectors
        edge_normal[i].XX =  d.YY / edgeLengths[i];
        edge_normal[i].YY = -1 * d.XX / edgeLengths[i]; // / (edgeLengths[i] * edgeLengths[i]);    
    }

    for (size_t i = 0; i < n; ++i) 
    {
        int j = i - 1;
        if (i == 0)
        {
            j = n - 1;
        }

        Vector2 vertex_normal = 0.5 * ((edge_normal[i] * edgeLengths[i]) + (edge_normal[j] * edgeLengths[j]));

        osmoticForce[i] = Kv * ((1.0/volume) - (1.0/V_bar)) * vertex_normal;
        // std::clog<<"Osmotic forces [" << i << "], ["<< j << "]: "<<osmoticForce[i].XX << ", " << osmoticForce[i].YY << std::endl;
    }

  return osmoticForce;
}


// Calculate the forces from mechanical surface tension acting on each vertex of the polygon
std::vector<Vector2> SpaceDynPolygon::calculateTensionForces() const {
  double Ksg = prop()->tension;
  const auto &vertex_positions = poly_.pts_;

  std::vector<Vector2> tensionForce(poly_.nbPoints());
  tensionForce.assign(poly_.nbPoints(), Vector2(0,0));

  Vector2 d, edgeUnitVector;
  real edgeLength;
  for (size_t i = 0; i < poly_.nbPoints(); ++i) {
    d.XX = vertex_positions[(i + 1)].xx - vertex_positions[i].xx;
    d.YY = vertex_positions[(i + 1)].yy - vertex_positions[i].yy;
    edgeLength = d.norm();
    edgeUnitVector = d / edgeLength;

    tensionForce[i] += -2 * Ksg * edgeLength * (-1 * edgeUnitVector);
    if (i == poly_.nbPoints() - 1) {
      tensionForce[0] += -2 * Ksg * edgeLength * (edgeUnitVector);
    } else {
      tensionForce[i + 1] += -2 * Ksg * edgeLength * (edgeUnitVector);
    }
  }

  return tensionForce;
}

std::vector<Vector2> SpaceDynPolygon::calculateRegularizationForces() const {
  double Ksl = 20.0; // regularization strength in units of mN.um;
  const auto &vertex_positions = poly_.pts_;

  size_t n = poly_.nbPoints();

  std::vector<Vector2> regForce(n);
  regForce.assign(n, Vector2(0,0));

  Vector2 d;
  std::vector<Vector2>edgeUnitVector(n);
  std::vector<real> edgeLength(n);
  real referenceLength = 0.0;

  for (size_t i = 0; i < n; ++i) {
    d.XX = vertex_positions[(i + 1)].xx - vertex_positions[i].xx;
    d.YY = vertex_positions[(i + 1)].yy - vertex_positions[i].yy;
    edgeLength[i] = d.norm();
    edgeUnitVector[i] = d / edgeLength[i];
    referenceLength += edgeLength[i];
  }

  int cnt = edgeLength.size();
  referenceLength = referenceLength / cnt; // average edge length 

  for (size_t i = 0; i < n; ++i) {

    regForce[i] += -2 * Ksl * (-1 * edgeUnitVector[i]) * (edgeLength[i] - referenceLength) / (referenceLength * referenceLength);
    if (i == n - 1) {
        regForce[0] += -2 * Ksl * edgeUnitVector[i] * (edgeLength[i] - referenceLength) / (referenceLength * referenceLength);
    } else {
        regForce[i + 1] += -2 * Ksl * edgeUnitVector[i] * (edgeLength[i] - referenceLength) / (referenceLength * referenceLength);
    }
    // std::clog << "Regularization forces: "<< regForce[i].XX << ", " << regForce[i].YY << std::endl;
 }

  return regForce;
}

std::vector<Vector2> SpaceDynPolygon::calculateBendingForces_optimized() const
{
    const real Kb = prop()->bending / 4;
    const size_t n = poly_.nbPoints();

    std::vector<Vector2> bendingForce(n, Vector2(0,0));
    if ( n < 3 )
        return bendingForce;

    std::vector<real> L(n), ang(n), T(n), c(n);
    std::vector<Vector2> u(n), nrm(n);

    for ( size_t i = 0; i < n; ++i )
    {
        real dx = poly_.pts_[(i+1)%n].xx - poly_.pts_[i].xx;
        real dy = poly_.pts_[(i+1)%n].yy - poly_.pts_[i].yy;
        L[i] = std::sqrt(dx*dx + dy*dy);
        u[i] = Vector2(dx/L[i], dy/L[i]);
        nrm[i] = Vector2(dy/(L[i]*L[i]), -dx/(L[i]*L[i]));
        ang[i] = std::atan2(dy, dx);
    }

    for ( size_t i = 0; i < n; ++i )
    {
        const size_t j = (i + n - 1) % n;               // previous edge
        real a = std::fmod(ang[j] - ang[i], 2*M_PI);
        a = std::fmod(a + M_PI, 2*M_PI) - M_PI;         // wrap to (-pi, pi]
        T[i] = std::tan(a/2);
        const real sec = 1 / std::cos(a/2);
        c[i] = 0.5 * sec * sec;
    }

    for ( size_t i = 0; i < n; ++i )
    {
        const size_t im = (i + n - 1) % n;
        const size_t i1 = (i + 1) % n;
        const size_t i2 = (i + 2) % n;

        const real kap = ( T[i] + T[i1] ) / L[i];
        const real g   = 2 * kap;                       // dE/dS, less Kb
        const real k2  = kap * kap;                     // dE/d(1/L), less Kb

        bendingForce[im] += -Kb * ( g * c[i] * nrm[im] );

        bendingForce[i]  += -Kb * ( g * ( -c[i]*nrm[im] - c[i]*nrm[i] + c[i1]*nrm[i] )
                                    + k2 * u[i] );

        bendingForce[i1] += -Kb * ( g * ( c[i]*nrm[i] - c[i1]*nrm[i] - c[i1]*nrm[i1] )
                                    - k2 * u[i] );

        bendingForce[i2] += -Kb * ( g * c[i1] * nrm[i1] );
    }
    return bendingForce;
}




void SpaceDynPolygon::update()
{
    surface_ = poly_.surface();
    if ( surface_ < 0 )
    {
        //std::clog << "flipping clockwise polygon `" << file << "'" << '\n';
        poly_.flip();
        surface_ = poly_.surface();
    }
    assert_true( surface_ > 0 );

    if ( poly_.complete(REAL_EPSILON) )
        throw InvalidParameter("unfit polygon: consecutive points may overlap");

    real box[4];
    poly_.find_extremes(box);
    inf_.set(box[0], box[2], -height_);
    sup_.set(box[1], box[3],  height_);

}

real SpaceDynPolygon::meanEdge() const {
  const unsigned n = poly_.nbPoints();
  if (n < 2)
    return 0;
  real s = 0;
  for (unsigned i = 0; i < n; ++i) {
    real dx = poly_.pts_[i + 1].xx - poly_.pts_[i].xx;
    real dy = poly_.pts_[i + 1].yy - poly_.pts_[i].yy;
    s += std::sqrt(dx * dx + dy * dy);
  }
  return s / n;
}


void SpaceDynPolygon::step() {
  
  std::vector<Vector2> force(
        calculateTensionForces() + calculateBendingForces_optimized()
        + vertex_forces + calculateRegularizationForces() + calculateOsmoticForces());

  for (unsigned i = 0; i < poly_.nbPoints(); ++i) {
      poly_.pts_[i].xx += prop()->mobility_dt * force[i].XX;
      poly_.pts_[i].yy += prop()->mobility_dt * force[i].YY;
    }

  poly_.wrap();
  update();

  reset_forces();
}


bool SpaceDynPolygon::inside(Vector const& W) const
{
#if ( DIM > 2 )
    if ( abs_real(W.ZZ) > height_ )
        return false;
#endif
#if ( DIM > 1 )
    return poly_.inside(W.XX, W.YY, 1);
#else
    return false;
#endif
}


Vector SpaceDynPolygon::place() const
{
    if ( surface_ <= 0 )
        throw InvalidParameter("cannot pick point inside polygon of null surface");
    return Space::place();
}


Vector SpaceDynPolygon::project(Vector const& W) const
{
    Vector P(W);
#if ( DIM == 2 )
    
    unsigned hit;
    poly_.project(W.XX, W.YY, P.XX, P.YY, hit);
    
#elif ( DIM > 2 )
    
    if ( abs_real(W.ZZ) > height_ )
    {
        if ( poly_.insideWinding(W.XX, W.YY) )
        {
            // too high or too low in the Z axis, but inside XY
            P.XX = W.XX;
            P.YY = W.YY;
        }
        else
        {
            // outside in Z and XY
            unsigned hit;
            poly_.project(W.XX, W.YY, P.XX, P.YY, hit);
        }
        P.ZZ = std::copysign(height_, W.ZZ);
    }
    else
    {
        unsigned hit;
        poly_.project(W.XX, W.YY, P.XX, P.YY, hit);
        if ( poly_.insideWinding(W.XX, W.YY) )
        {
            // inside in the Z axis and the XY polygon:
            // to the polygonal edge in XY plane:
            real HH = (W.XX-P.XX)*(W.XX-P.XX) + (W.YY-P.YY)*(W.YY-P.YY);
            // to the top/bottom plates:
            real V = height_ - abs_real(W.ZZ);
            // compare distances
            if ( V * V < HH )
                return Vector(W.XX, W.YY, std::copysign(height_, W.ZZ));
        }
        P.ZZ = W.ZZ;
    }
    
#endif
    return P;
}

real SpaceDynPolygon::depthBelowMobileEdge(Vector const& pos) const
{
    const real d = ( pos - project(pos) ).norm();
    return inside(pos) ? d : -d;
}


Vector SpaceDynPolygon::normalToEdge(Vector const& pos) const
{
    Vector prj = project(pos);
    Vector d = pos - prj;
    real n2 = d.normSqr();

    if ( n2 > square(1024*REAL_EPSILON) )
    {
        Vector u = d / std::sqrt(n2);
        return inside(pos) ? -u : u;        // outward
    }

    unsigned best = 0;
    real bestDa = INFINITY;
    Vector2 P(pos.XX, pos.YY);
    for ( unsigned j = 0; j < poly_.nbPoints(); ++j )
    {
        Vector2 b;
        real da;
        edgeProjection(j, P, b, da);
        if ( da < bestDa ) { bestDa = da; best = j; }
    }

    // update() keeps the polygon anti-clockwise, for which (dy, -dx) is outward
    Vector n(poly_.pts_[best].dy, -poly_.pts_[best].dx, 0);
    if ( inside(prj + 0.0001 * n) )
        n = -n;
    return n;
}


//------------------------------------------------------------------------------
#pragma mark - setConfinement

real SpaceDynPolygon::edgeProjection(unsigned j, Vector2 const& P,
                                     Vector2& b, real& da) const
{
    real x = P.XX - poly_.pts_[j].xx;
    real y = P.YY - poly_.pts_[j].yy;

    // abscissa of the projection on segment [j, j+1], clamped to the segment
    real a = poly_.pts_[j].dx * x + poly_.pts_[j].dy * y;
    a = std::min(poly_.pts_[j].len, std::max(real(0), a));

    b.XX = x - a * poly_.pts_[j].dx;
    b.YY = y - a * poly_.pts_[j].dy;
    da = b.norm();
    return a;
}


/**
 Force on the filament point from the membrane and vice versa
 */
void SpaceDynPolygon::setConfinement(Vector const&pos, Mecapoint const& mp,
                                         Meca& meca, real stiff) const
{
    const real cutoff = prop()->cutoff;
    const size_t npoints = poly_.nbPoints();
    Vector2 P(pos.XX, pos.YY);

    // only points that have escaped the polygon are pushed back
    if ( poly_.insideWinding(P.XX, P.YY) != 0 )
        return;

    // closest point of the whole membrane
    unsigned best = npoints;
    real bestDa = INFINITY, bestA = 0;
    Vector2 bestB(0,0);
    for ( unsigned j = 0; j < npoints; ++j )
    {
        Vector2 b;
        real da;
        real a = edgeProjection(j, P, b, da);
        if ( da < bestDa ) { bestDa = da; bestB = b; bestA = a; best = j; }
    }
    if ( best >= npoints || bestDa <= 0 )
        return;

    meca.addForce(mp, -stiff * Vector(bestB.XX, bestB.YY, 0));

    if ( bestDa < cutoff )
    {
        const real len = poly_.pts_[best].len;
        const unsigned k = ( best + 1 == npoints ) ? 0 : best + 1;

        decomposeForce(((len - bestA) / len) * stiff * bestB, best);
        decomposeForce((bestA / len) * stiff * bestB, k);
    }
}


//------------------------------------------------------------------------------
#pragma mark - I/O

void SpaceDynPolygon::write(Outputter &out) const {
    
    writeMarker(out, TAG);
    writeShape(out, "dynamic_polygon");
    
    // out.put_characters("dynaPoly", 16);
    // out.writeUInt8(poly_.closed);
    
    out.writeUInt32(poly_.nbPoints());
    for (int i = 0; i < poly_.nbPoints(); i++) {
        out.writeFloat(poly_.pts_[i].xx);
        out.writeFloat(poly_.pts_[i].yy);
        // currently don't print out info
        // out.writeInt32(poly_.pts_[i].info);
    }
}

void SpaceDynPolygon::read(Inputter &in, Simul &, ObjectTag) {
    
    std::string str, str2;
    str = in.get_characters(16); // stored as 16 characters
    // str2 = in.get_characters(16);

    // check that this matches current Space:
    if (str.compare(0, 15, "dynamic_polygon")) {
        std::ostringstream oss;
        oss << "Space `" << prop()->name() << "' has shape " << str;
        oss << " in objects and " << prop()->shape << " in property";
        throw InvalidIO(oss.str());
    }
    // poly_.closed = in.readUInt8();
    // n is the total number of points in the polygon
    poly_.npts_ = in.readUInt32();
    poly_.allocate(poly_.npts_);
    unsigned int n = poly_.npts_;
    // Read in 2*n corresponding to x1, y1, x2, y2 .. xn, yn
    for (int i = 0; i < poly_.npts_; ++i) {
        poly_.pts_[i].xx = in.readFloat();
        poly_.pts_[i].yy = in.readFloat();

    }

    update();
    resize_force(n); // resize forces to fit number of polygon vertices
}

void SpaceDynPolygon::setLengths(const real len[8])
{
    height_ = len[0];
}

void SpaceDynPolygon::report(std::ostream& os) const
{
    // enclosed area as the membrane itself sees it: this is the `volume' that
    // drives the osmotic term against V_bar, so it is worth being able to read
    os << " nbPoints " << poly_.nbPoints() << " surface " << poly_.surface();
    for (int i = 0; i < poly_.nbPoints(); i++) {
        os << "\n" << poly_.pts_[i].xx <<"\t" << poly_.pts_[i].yy;
        // currently don't print out info
        // out.writeInt32(poly_.pts_[i].info);
    }
}

//------------------------------------------------------------------------------
#pragma mark - OpenGL display

#ifdef DISPLAY

#include "gle.h"
#include "gym_flute.h"
#include "gym_draw.h"

void SpaceDynPolygon::drawPolygon(float lines, float points) const
{
    const unsigned nbp = poly_.nbPoints();
    Polygon::Point2D const* pts = poly_.pts_;
    flute2 * flt = gym::mapBufferV2(nbp+1);
    for ( size_t n = 0; n <= nbp; ++n )
    {
        std::clog << "flute2["<<n<<"] = "<<pts[n].xx<<", "<<pts[n].yy<<std::endl;
        flt[n].set(pts[n].xx, pts[n].yy);
    }
    gym::unmapBufferV2();
    
    glEnable(GL_STENCIL_TEST);
    glClearStencil(1);
    glClear(GL_STENCIL_BUFFER_BIT);
    glStencilFunc(GL_EQUAL, 1, ~0U);
    glStencilOp(GL_KEEP, GL_ZERO, GL_ZERO);
    if ( lines > 0 )
    {
        gym::drawLineStrip(lines, 0, nbp+1);
    }
    if ( points > 0 )
    {
        gym::drawPoints(points, 0, nbp);
    }
    glClear(GL_STENCIL_BUFFER_BIT);
    glDisable(GL_STENCIL_TEST);
}

void SpaceDynPolygon::draw3D() const
{
    const float H(-height_);
    const unsigned nbp = poly_.nbPoints();
    Polygon::Point2D const* pts = poly_.pts_;
    flute3 * flt = gym::mapBufferV3(2*nbp+2);
    for ( size_t i = 0; i <= nbp; ++i )
    {
        float X(pts[i].xx), Y(pts[i].yy);
        flt[2*i  ] = { X, Y, -H };
        flt[2*i+1] = { X, Y,  H };
    }
    gym::unmapBufferV3();
    // display sides
    gym::drawTriangleStrip(0, 2*nbp+2);

    float lines = 2;
    if ( lines )
    {
        // display bottom
        gym::rebindBufferV3(2, 0);
        gym::drawLineStrip(lines, 0, nbp+1);
        // display top
        gym::rebindBufferV3(2, 1);
        gym::drawLineStrip(lines, 0, nbp+1);
    }
    gym::cleanupV();
}

#else

void SpaceDynPolygon::drawPolygon(float, float) const {}
void SpaceDynPolygon::draw3D() const {}

#endif

//------------------------------------------------------------------------------
#pragma mark - Energies (must match the force kernels above)

/**
 E = Ksg * sum_i L_i^2
 */
real SpaceDynPolygon::energyTension() const
{
    const real Ksg = prop()->tension;
    const size_t n = poly_.nbPoints();
    real E = 0;
    for ( size_t i = 0; i < n; ++i )
    {
        real dx = poly_.pts_[(i+1)%n].xx - poly_.pts_[i].xx;
        real dy = poly_.pts_[(i+1)%n].yy - poly_.pts_[i].yy;
        E += Ksg * ( dx*dx + dy*dy );
    }
    return E;
}


/**
 E = Ksl * sum_i (L_i - L_mean)^2 / L_mean^2,
 */
real SpaceDynPolygon::energyRegularization() const
{
    const real Ksl = 20.0;
    const size_t n = poly_.nbPoints();
    std::vector<real> L(n);
    real ref = 0;
    for ( size_t i = 0; i < n; ++i )
    {
        real dx = poly_.pts_[(i+1)%n].xx - poly_.pts_[i].xx;
        real dy = poly_.pts_[(i+1)%n].yy - poly_.pts_[i].yy;
        L[i] = std::sqrt(dx*dx + dy*dy);
        ref += L[i];
    }
    ref /= n;
    real E = 0;
    for ( size_t i = 0; i < n; ++i )
        E += Ksl * (L[i]-ref) * (L[i]-ref) / ( ref * ref );
    return E;
}


/**
 E = Kb * sum_i kappa_i^2 L_i,  kappa_i = (tan(t_i/2) + tan(t_{i+1}/2)) / L_i,
 with Kb = bending/4 and t_i the turning angle at vertex i.
 */
real SpaceDynPolygon::energyBending() const
{
    const real Kb = prop()->bending / 4;
    const size_t n = poly_.nbPoints();
    std::vector<real> L(n), ang(n), tanHalf(n);

    for ( size_t i = 0; i < n; ++i )
    {
        real dx = poly_.pts_[(i+1)%n].xx - poly_.pts_[i].xx;
        real dy = poly_.pts_[(i+1)%n].yy - poly_.pts_[i].yy;
        L[i] = std::sqrt(dx*dx + dy*dy);
        ang[i] = std::atan2(dy, dx);
    }
    for ( size_t i = 0, j = n-1; i < n; ++i, ++j )
    {
        if ( j == n ) j = 0;
        real a = std::fmod(ang[j] - ang[i], 2*M_PI);
        a = std::fmod(a + M_PI, 2*M_PI) - M_PI;
        tanHalf[i] = std::tan(a/2);
    }
    real E = 0;
    for ( size_t i = 0, j = 1; i < n; ++i, ++j )
    {
        if ( j == n ) j = 0;
        real k = ( tanHalf[i] + tanHalf[j] ) / L[i];
        E += Kb * k * k * L[i];
    }
    return E;
}


/**
 van 't Hoff form, E = Kv * ( V/V_bar - 1 - log(V/V_bar) ).
 */
real SpaceDynPolygon::energyOsmotic() const
{
    const real Kv = prop()->Kv;
    const real V_bar = prop()->V_bar;
    if ( Kv == 0 )
        return 0;               // matches the short-circuit in the force kernel
    real V = abs_real(poly_.surface());
    if ( V <= 0 )
        return 0;
    return Kv * ( V/V_bar - 1 - std::log(V/V_bar) );
}


real SpaceDynPolygon::totalEnergy() const
{
    return energyTension() + energyBending()
         + energyRegularization() + energyOsmotic();
}


/**
 Finite-difference audit: for each term, compare -dE/dx against the force the
 matching force calculation produces.
 */
void SpaceDynPolygon::selfTest(std::ostream& os) const
{
    const size_t n = poly_.nbPoints();
    if ( n < 3 ) { os << "\npolygon too small for selfTest"; return; }

    struct Term {
        const char* name;
        real (SpaceDynPolygon::*energy)() const;
        std::vector<Vector2> (SpaceDynPolygon::*force)() const;
    };
    const Term terms[] = {
        { "tension",        &SpaceDynPolygon::energyTension,        &SpaceDynPolygon::calculateTensionForces },
        { "bending(opt)",   &SpaceDynPolygon::energyBending,        &SpaceDynPolygon::calculateBendingForces_optimized },
        { "regularization", &SpaceDynPolygon::energyRegularization, &SpaceDynPolygon::calculateRegularizationForces },
        { "osmotic",        &SpaceDynPolygon::energyOsmotic,        &SpaceDynPolygon::calculateOsmoticForces },
    };

    // step scaled to the geometry, small enough for a centred difference
    real h = 1e-6 * meanEdge();
    if ( h <= 0 ) h = 1e-9;

    auto rewrap = [&]() {
        poly_.pts_[n]   = poly_.pts_[0];
        poly_.pts_[n+1] = poly_.pts_[1];
    };

    std::streamsize prec = os.precision(4);
    os << std::scientific;
    os << "\n% finite-difference check, h = " << h << ", " << n << " vertices";
    os << "\n% term              max|F_analytic - F_fd|   max|F_analytic|   rel";

    for ( Term const& t : terms )
    {
        std::vector<Vector2> F = (this->*t.force)();
        real worst = 0, scale = 0;
        for ( size_t i = 0; i < n; ++i )
        {
            for ( int c = 0; c < 2; ++c )
            {
                real& x = c ? poly_.pts_[i].yy : poly_.pts_[i].xx;
                const real x0 = x;
                x = x0 + h; rewrap();
                const real Ep = (this->*t.energy)();
                x = x0 - h; rewrap();
                const real Em = (this->*t.energy)();
                x = x0;     rewrap();

                const real fd = -( Ep - Em ) / ( 2 * h );
                const real fa = c ? F[i].YY : F[i].XX;
                worst = std::max(worst, abs_real(fa - fd));
                scale = std::max(scale, abs_real(fa));
            }
        }
        os << "\n  " << t.name;
        for ( size_t k = std::string(t.name).size(); k < 18; ++k ) os << ' ';
        os << worst << "   " << scale << "   ";
        os << ( scale > 0 ? worst/scale : worst );
    }
    os.unsetf(std::ios_base::floatfield);
    os.precision(prec);
}
