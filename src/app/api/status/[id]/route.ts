import { NextResponse } from 'next/server';

export async function GET(
    request: Request,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const { id } = await params;

        // Proxy the request to FastAPI
        const response = await fetch(`http://localhost:8000/status/${id}`, {
            method: 'GET',
        });

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Status check error:', error);
        return NextResponse.json({ error: 'Failed to check status' }, { status: 500 });
    }
}
