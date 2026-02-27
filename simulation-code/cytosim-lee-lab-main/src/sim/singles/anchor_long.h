// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef ANCHOR_LONG_H
#define ANCHOR_LONG_H

#include "anchor.h"
#include "object_set.h"


class AnchorLong : public Anchor
{
    
    /// the side (top/bottom) of the interaction
    mutable Torque mArm;
    
    /// used to recalculate `mArm`
    static Torque calcArm(Interpolation const& pt, Vector const& pos, real len);
    
public:

    /// constructor
    AnchorLong(SingleProp const*, Vector const& = Vector(0,0,0));

    /// destructor
    ~AnchorLong();
    
    /// recalculates `mArm` when making a bridge
    void afterAttachment(Hand const*);

    /// position on the side of fiber used in setInteractions()
    Vector sidePos() const;
    
    /// force = stiffness * ( posFoot() - posHand() )
    Vector force() const;
    
    /// Monte-Carlo step if Hand is attached
    void stepA();

    /// add interactions to a Meca
    void setInteractions(Meca&) const;
    
};


#endif
