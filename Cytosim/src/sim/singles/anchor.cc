// Cytosim was created by Francois Nedelec. Copyright 2020 Cambridge University

#include "anchor.h"
#include "space.h"
#include "object_set.h"
#include "simul.h"
#include "meca.h"
#include "modulo.h"



Anchor::Anchor(SingleProp const* p, Vector const& w)
: Single(p, w)
{
#if ( 0 )
    if ( p->diffusion > 0 )
        throw InvalidParameter(name()+":diffusion cannot be > 0 if activity=fixed");
#endif
}


Anchor::~Anchor()
{
    //std::clog<<"~Anchor("<<this<<")\n";
}


void Anchor::beforeDetachment(Hand const*)
{
    assert_true( attached() );

    SingleSet * set = static_cast<SingleSet*>(objset());
    if ( set )
        set->relinkD(this);
}


void Anchor::stepF()
{
    assert_false( sHand->attached() );

    sHand->stepUnattached(simul(), sPos);
}


void Anchor::stepA()
{
    assert_true( sHand->attached() );
    
    Vector f = Anchor::force();
    Fiber const* fib = fiber();


    if ( sHand->checkKramersDetachment(f.norm()) )
    {
        sHand->detach();
        fib->objset()->eraseObject(const_cast<Fiber*>(fib));
        
        return;
    }
    else
        sHand->stepLoaded(f);
    
}


/**
 This calculates the force corresponding to addPointClamp()
 */
Vector Anchor::stretch() const
{
    assert_true( sHand->attached() );
    Vector d = sPos - posHand();
    
    if ( modulo )
        modulo->fold(d);
    
    return d;
}

/**
 This calculates the force corresponding to addPointClamp()
 */
Vector Anchor::force() const
{
    assert_true( sHand->attached() );
    Vector d = sPos - posHand();
    
    if ( modulo )
        modulo->fold(d);
    
    return prop->stiffness * d;
}


void Anchor::setInteractions(Meca& meca) const
{
    assert_true( prop->length == 0 );
    meca.addPointClamp(sHand->interpolation(), sPos, prop->stiffness);
}
