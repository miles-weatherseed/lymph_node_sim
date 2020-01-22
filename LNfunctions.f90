subroutine inside_sphere(vec, radius, sts)

    implicit none
    double precision, dimension(3), intent(in) :: vec
    double precision, intent(in) :: radius
    logical, intent(out) :: sts

    if (NORM2(vec) < radius) then
        sts = .TRUE.
    else
        sts = .FALSE.
    end if

end subroutine inside_sphere

subroutine threed_to_1d(coord_vec, numPositions, oned)

    implicit none
    integer, dimension(3), intent(in) :: coord_vec
    integer, intent(in) :: numPositions
    integer, intent(out) :: oned

    oned = coord_vec(1)*numPositions**2 + coord_vec(2)*numPositions + coord_vec(3)

end subroutine threed_to_1d

subroutine set_coordinates(posn_vec, radius, cellSide, coord_vec)

    implicit none
    double precision, dimension(3), intent(in) :: posn_vec
    double precision, intent(in) :: radius
    double precision, intent(in) :: cellSide
    integer, dimension(3), intent(out) :: coord_vec

    coord_vec = INT((posn_vec + radius)/cellSide)

end subroutine set_coordinates

subroutine return_to_sphere(current_posn, new_posn_vec, vmag, freePathMean, freePathStDev, output_posn_vec)

    use randgen
    double precision, dimension(3), intent(in) :: current_posn, new_posn_vec
    double precision, intent(in) :: vmag, freePathMean, freePathStDev
    double precision, dimension(3), intent(out) :: output_posn_vec
    double precision, dimension(3) :: velocity
    double precision :: theta, phi, gamma
    integer :: i

    theta = ACOS(RAND(1))
    phi = 2*3.1415926535*RAND(1)
    current_posn = current_posn/NORM2(current_posn)
    velocity = -vmag*current_posn


end subroutine return_to_sphere