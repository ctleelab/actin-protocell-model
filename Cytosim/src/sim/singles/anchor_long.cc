// Cytosim was created by Francois Nedelec. Copyright 2021 Cambridge University
#include "anchor_long.h"
#include "space.h"
#include "fiber.h"
#include "simul_part.h"
#include "meca.h"
#include "modulo.h"



AnchorLong::AnchorLong(SingleProp const* p, Vector const& w)
: Anchor(p, w), mArm(nullTorque)
{
#if ( 0 )
    if ( p->diffusion > 0 )
        throw InvalidParameter(name()+":diffusion cannot be > 0 if activity=fixed");
#endif
}


AnchorLong::~AnchorLong()
{
    //std::clog<<"~AnchorLong("<<this<<")\n";
}

//------------------------------------------------------------------------------

Torque AnchorLong::calcArm(Interpolation const& pt, Vector const& pos, real len)
{
    Vector off = pt.pos1() - pos;
    if ( modulo )
        modulo->fold(off);
#if ( DIM >= 3 )
    off = cross(off, pt.diff());
    real n = off.norm();
    if ( n > REAL_EPSILON )
        return off * ( len / n );
    else
        return pt.dir().randOrthoU(len);
#else
    return std::copysign(len, cross(off, pt.diff()));
#endif
}


void AnchorLong::afterAttachment(Hand const* ha)
{
    Single::afterAttachment(ha);

#if ( DIM > 1 )
    Interpolation const& ipt = sHand->interpolation();
    mArm = calcArm(ipt, sPos, prop->length);
#endif
}

/*
 Note that, since `mArm` is calculated by setInteractions(),
 the result will be incorrect if 'solve=0'
*/
Vector AnchorLong::sidePos() const
{
#if ( DIM > 1 )
    return sHand->pos() + cross(mArm, sHand->dirFiber());
#else
    return sHand->pos();
#endif
}


Vector AnchorLong::force() const
{
    assert_true( sHand->attached() );
    Vector d = sPos - AnchorLong::sidePos();
 
    if ( modulo )
        modulo->fold(d);
    
    return prop->stiffness * d;
}


void AnchorLong::stepA()
{
    assert_true( sHand->attached() );
    assert_true( hasLink() );

    Vector f = AnchorLong::force();
    Fiber const* fib = fiber();

    if ( sHand->checkKramersDetachment(f.norm()) )
    {
        sHand->detach();
        fib->objset()->eraseObject(const_cast<Fiber*>(fib));
        
        return;
    }
   
    else
        sHand->stepLoaded(f);
    
    Vector pos = position() + Vector::randS(prop->diffusion_dt);

    // confinement:
    if ( prop->confine == CONFINE_INSIDE )
    {
        sPos = prop->confine_space->bounce(pos);
    }
    else if ( prop->confine == CONFINE_ON )
    {
        sPos = prop->confine_space->project(pos);
    }
    else
    {
        sPos = pos;
    }

    setPosition(sPos);
    
}


void AnchorLong::setInteractions(Meca& meca) const
{
#if ( DIM == 1 )
    meca.addPointClamp(sHand->interpolation(), sPos, prop->stiffness);
#else
    Interpolation const& ipt = sHand->interpolation();
    
    /* 
     The 'arm' is recalculated every time, but in 2D at least,
     this may not be necessary, as flipping should occur rarely.
     */
    
    mArm = calcArm(ipt, sPos, prop->length);
    
#if ( DIM == 2 )
    meca.addSidePointClamp2D(ipt, sPos, mArm, prop->stiffness);
#elif ( DIM >= 3 )
    meca.addSidePointClamp3D(ipt, sPos, mArm, prop->stiffness);
#endif

#endif

}


